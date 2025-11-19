import pandas as pd
from datasets import Dataset

# 1. Cargar los archivos JSONL
# Asegúrate de que las rutas a tus archivos sean correctas
df_judgments = pd.read_json("./just-nlp-folders/datasets/train/train_judg.jsonl", lines=True)
df_summaries = pd.read_json("./just-nlp-folders/datasets/train/train_ref_summ.jsonl", lines=True)

# 2. Unir los DataFrames basándonos en la columna "ID"
# Esto crea una sola fila con el Judgment y su Summary correspondiente
df_merged = pd.merge(df_judgments, df_summaries, on="ID")

# 3. Definir el Prompt del Sistema (Developer)
# Esto le dice al modelo cómo comportarse.
# Dejamos el prompt ganador
SYSTEM_PROMPT = """You are a legal expert AI. Your task is to provide concise and accurate summaries of legal judgments.
Focus on the key facts, the legal reasoning, and the final verdict."""

# 4. Función para crear la estructura exacta de 'messages'
def create_messages(row):
    # El formato que unsloth y gpt-oss esperan es una lista de diccionarios
    # Nota: Como no tienes datos de "thinking" (análisis), lo dejamos en null.
    
    return [
        {
            "role": "system",
            "content": "reasoning language: English\n\n" + SYSTEM_PROMPT,
            "thinking": None 
        },
        {
            "role": "user",
            "content": row["Judgment"],
            "thinking": None
        },
        {
            "role": "assistant",
            "content": row["Summary"],
            # Aquí normalmente iría el 'Analysis' si lo tuvieras. 
            # Al ponerlo null, haces un fine-tuning directo (Input -> Output).
            "thinking": None 
        }
    ]

# 5. Aplicar la función a cada fila
df_merged["messages"] = df_merged.apply(create_messages, axis=1)

# 6. Seleccionar solo la columna 'messages' (es la única que necesitas para entrenar)
final_df = df_merged[["messages"]]

# 7. Convertir a objeto Dataset de Hugging Face
dataset = Dataset.from_pandas(final_df)

# --- OPCIÓN A: Guardar en disco local ---
# dataset.to_json("train_dataset_final.jsonl")
# print("Archivo guardado como 'train_dataset_final.jsonl'")

# --- OPCIÓN B: Subir directamente a Hugging Face ---
# Descomenta las siguientes lineas si ya estás logueado con `huggingface-cli login`
# dataset.push_to_hub("tu_usuario/nombre_del_dataset")

dataset.push_to_hub("andrewmos/indian-legal-summaries", token="...")