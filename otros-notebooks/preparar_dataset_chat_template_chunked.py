import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# ================================================================
# 1. CONFIGURACIÓN
# ================================================================

MODEL_NAME = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
HF_DATASET_NAME = "andrewmos/indian-legal-summaries-chat-chunked"
HF_TOKEN = "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"   # ← coloca tu token aquí

CHUNK_SIZE = 6500     # tokens por chunk
OVERLAP = 200         # tokens de solapamiento

# ================================================================
# 2. CARGAR TOKENIZER
# ================================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ================================================================
# 3. FUNCIÓN DE CHUNKING (TOKEN-LEVEL)
# ================================================================

def chunk_text(text, tokenizer, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    ids = tokenizer(text)["input_ids"]
    chunks = []

    start = 0
    while start < len(ids):
        end = start + chunk_size
        chunk_ids = ids[start:end]
        chunk_txt = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_txt)
        start += chunk_size - overlap

    return chunks

# ================================================================
# 4. CARGAR Y UNIR TUS JSONL ORIGINALES
# ================================================================

df_j = pd.read_json("./just-nlp-folders/datasets/train/train_judg.jsonl", lines=True)
df_s = pd.read_json("./just-nlp-folders/datasets/train/train_ref_summ.jsonl", lines=True)

df = pd.merge(df_j, df_s, on="ID")

print("Total de juicios originales:", len(df))

# ================================================================
# 5. CHUNKEAR TODO EL DATASET
# ================================================================

rows = []

for _, row in df.iterrows():
    chunks = chunk_text(row["Judgment"], tokenizer)

    for i, chunk in enumerate(chunks):
        rows.append({
            "orig_id": row["ID"],
            "ID": f"{row['ID']}_chunk{i}",
            "chunk": chunk,
            "summary": row["Summary"],
        })

df_chunked = pd.DataFrame(rows)
print("Total de chunks generados:", len(df_chunked))

# ================================================================
# 6. CONSTRUIR messages ESTILO GEMMA-3
# ================================================================

INSTRUCTION = (
    "Provide a concise and accurate summary of the following legal judgment. "
    "Focus on the key facts, the legal reasoning, and the final verdict."
)

def build_messages(chunk, summary):
    user_msg = f"{INSTRUCTION}\n\n---\n\n{chunk}"
    return [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": summary}
    ]

df_chunked["messages"] = df_chunked.apply(
    lambda r: build_messages(r["chunk"], r["summary"]),
    axis=1
)

# ================================================================
# 7. SPLIT 80/20 (YA CON CHUNKS)
# ================================================================

train_df, test_df = train_test_split(
    df_chunked,
    test_size=0.20,
    shuffle=True,
    random_state=42
)

print("Train:", len(train_df))
print("Test :", len(test_df))

# ================================================================
# 8. CONVERTIR A HUGGINGFACE DATASET
# ================================================================

train_dataset = Dataset.from_pandas(
    train_df[["ID", "messages"]],
    preserve_index=False
)

dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# ================================================================
# 9. SUBIR AL HUGGINGFACE HUB
# ================================================================

dataset_dict.push_to_hub(
    HF_DATASET_NAME,
    token=HF_TOKEN
)

print("\n🚀 Dataset CHUNKEADO subido correctamente!")
print("📦 Nombre en HuggingFace:", HF_DATASET_NAME)