import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# 1. Cargar los archivos JSONL
df_judgments = pd.read_json("./just-nlp-folders/datasets/train/train_judg.jsonl", lines=True)
df_summaries = pd.read_json("./just-nlp-folders/datasets/train/train_ref_summ.jsonl", lines=True)

# 2. Unir por ID
df_merged = pd.merge(df_judgments, df_summaries, on="ID")

# 3. Instrucción estilo Alpaca
INSTRUCTION = (
    "Provide a concise and accurate summary of the following legal judgment. "
    "Focus on the key facts, the legal reasoning, and the final verdict."
)

# 4. Dataset estilo Alpaca-cleaned
df_alpaca = pd.DataFrame({
    'id': df_merged['ID'].values,
    'instruction': [INSTRUCTION] * len(df_merged),
    'input': df_merged['Judgment'].values,
    'output': df_merged['Summary'].values,
})

print(f"Dataset creado con {len(df_alpaca)} ejemplos")

# 5. Hacer split 80/20 (train/test)
train_df, test_df = train_test_split(df_alpaca, test_size=0.20, shuffle=True, random_state=42)

print(f"Train: {len(train_df)} ejemplos")
print(f"Test: {len(test_df)} ejemplos")

# 6. Convertir a Dataset de Hugging Face
train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

# 7. Crear DatasetDict (estructura correcta para subir a HF)
dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# 8. Subir al Hugging Face Hub (sube ambos splits)
dataset_dict.push_to_hub(
    "andrewmos/indian-legal-summaries-alpaca-format",
    token=""
)

print("\n✅ Dataset con train/test subido exitosamente a HuggingFace Hub")
print("📊 Estadísticas:")
print(f"   - Train: {len(train_dataset)} ejemplos")
print(f"   - Test: {len(test_dataset)} ejemplos")
print(f"   - Promedio input chars: {df_alpaca['input'].str.len().mean():.1f}")
print(f"   - Promedio output chars: {df_alpaca['output'].str.len().mean():.1f}")
print(f"   - Columnas: {df_alpaca.columns.tolist()}")