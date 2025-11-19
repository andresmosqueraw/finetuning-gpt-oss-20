import pandas as pd
from datasets import Dataset

# 1. Cargar los archivos JSONL
# Asegúrate de que las rutas a tus archivos sean correctas
df_judgments = pd.read_json("./just-nlp-folders/datasets/train/train_judg.jsonl", lines=True)
df_summaries = pd.read_json("./just-nlp-folders/datasets/train/train_ref_summ.jsonl", lines=True)

# 2. Unir los DataFrames basándonos en la columna "ID"
# Esto crea una sola fila con el Judgment y su Summary correspondiente
df_merged = pd.merge(df_judgments, df_summaries, on="ID")

# 3. Definir la instrucción base (similar al formato alpaca-cleaned)
# Esta será la instrucción que se aplica a todos los ejemplos
INSTRUCTION = "Provide a concise and accurate summary of the following legal judgment. Focus on the key facts, the legal reasoning, and the final verdict."

# 4. Crear el dataset en formato alpaca-cleaned
# Formato: instruction, input, output
df_alpaca = pd.DataFrame({
    'instruction': [INSTRUCTION] * len(df_merged),
    'input': df_merged['Judgment'].values,
    'output': df_merged['Summary'].values
})

print(f"Dataset creado con {len(df_alpaca)} ejemplos")
print(f"\nColumnas: {df_alpaca.columns.tolist()}")
print(f"\nEjemplo del primer registro:")
print(f"Instruction: {df_alpaca.iloc[0]['instruction']}")
print(f"Input (primeros 200 caracteres): {df_alpaca.iloc[0]['input'][:200]}...")
print(f"Output (primeros 200 caracteres): {df_alpaca.iloc[0]['output'][:200]}...")

# 5. Convertir a objeto Dataset de Hugging Face
dataset = Dataset.from_pandas(df_alpaca)

# --- OPCIÓN A: Guardar en disco local ---
# dataset.to_json("train_dataset_alpaca_format.jsonl")
# print(f"\n✅ Archivo guardado como 'train_dataset_alpaca_format.jsonl'")

# También guardar como JSON para mayor compatibilidad
# dataset.to_json("train_dataset_alpaca_format.json")
# print(f"✅ Archivo guardado como 'train_dataset_alpaca_format.json'")

# --- OPCIÓN B: Subir directamente a Hugging Face ---
# Descomenta las siguientes líneas si ya estás logueado con `huggingface-cli login`
dataset.push_to_hub("andrewmos/indian-legal-summaries-alpaca-format", token="")
print(f"✅ Dataset subido a Hugging Face Hub")

print(f"\n📊 Estadísticas del dataset:")
print(f"   - Total de ejemplos: {len(dataset)}")
print(f"   - Longitud promedio de 'input': {df_alpaca['input'].str.len().mean():.1f} caracteres")
print(f"   - Longitud promedio de 'output': {df_alpaca['output'].str.len().mean():.1f} caracteres")
print(f"   - Características: {dataset.features}")

