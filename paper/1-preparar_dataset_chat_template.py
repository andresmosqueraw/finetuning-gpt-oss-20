import pandas as pd
from typing import List, Dict
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# ================================================================
# 1. CONFIGURACIÓN
# ================================================================

MODEL_NAME: str = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
HF_DATASET_NAME: str = "andrewmos/indian-legal-summaries-chat-template"
HF_TOKEN: str = ""
df_j = pd.read_json("./just-nlp-folders/datasets/train/train_judg.jsonl", lines=True)
df_s = pd.read_json("./just-nlp-folders/datasets/train/train_ref_summ.jsonl", lines=True)

# ================================================================
# 2. CARGAR TOKENIZER
# ================================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ================================================================
# 3. CARGAR Y UNIR JSONL ORIGINALES
# ================================================================

df_j = pd.read_json("./just-nlp-folders/datasets/train/train_judg.jsonl", lines=True)
df_s = pd.read_json("./just-nlp-folders/datasets/train/train_ref_summ.jsonl", lines=True)

df = pd.merge(df_j, df_s, on="ID")

print("Total de registros originales:", len(df))

# ================================================================
# 4. CONSTRUIR messages ESTILO GEMMA-3 (SIN CHUNKS)
# ================================================================

INSTRUCTION: str = (
    "Provide a concise and accurate summary of the following legal judgment. "
    "Focus on the key facts, the legal reasoning, and the final verdict."
)

def build_messages(full_judgment: str, summary: str) -> List[Dict[str, str]]:
    """
    Construye la estructura de mensajes para el fine-tuning de un modelo de chat.

    Esta función toma el texto completo de un juicio y su resumen correspondiente,
    y los formatea en una lista de diccionarios siguiendo el esquema de roles
    'user' y 'assistant'.

    Args:
        full_judgment (str): El texto completo del juicio legal.
        summary (str): El resumen del juicio (target).

    Returns:
        List[Dict[str, str]]: Una lista de mensajes donde cada mensaje es un diccionario
                              con las claves 'role' y 'content'.
    """
    user_msg = f"{INSTRUCTION}\n\n---\n\n{full_judgment}"
    return [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": summary}
    ]

df["messages"] = df.apply(
    lambda r: build_messages(r["Judgment"], r["Summary"]),
    axis=1
)

# ================================================================
# 5. SPLIT 80/20 (SIN CHUNKS)
# ================================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    shuffle=True,
    random_state=42
)

print("Train:", len(train_df))
print("Test :", len(test_df))

# ================================================================
# 6. CONVERTIR A HUGGINGFACE DATASET
# ================================================================

train_dataset = Dataset.from_pandas(
    train_df[["ID", "messages"]],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df[["ID", "messages"]],
    preserve_index=False
)

dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# ================================================================
# 7. SUBIR AL HUGGINGFACE HUB
# ================================================================

dataset_dict.push_to_hub(
    HF_DATASET_NAME,
    token=HF_TOKEN
)

print("📦 Nombre en HuggingFace:", HF_DATASET_NAME)
