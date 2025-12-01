import pandas as pd
import json
import os

JSONL_OUTPUT = "evaluacion_test_chunked_Finetuned_fast.jsonl"

print(f"📂 Leyendo archivo: {JSONL_OUTPUT} ...")

if not os.path.exists(JSONL_OUTPUT):
    print("⚠️ El archivo no existe todavía.")
    exit()

eval_data = []
with open(JSONL_OUTPUT, "r", encoding="utf-8") as f:
    for line in f:
        try:
            line = line.strip()
            if not line: continue
            eval_data.append(json.loads(line))
        except json.JSONDecodeError:
            # Ignorar líneas corruptas o incompletas si el proceso se interrumpió justo escribiendo
            pass

if not eval_data:
    print("⚠️ El archivo está vacío o no tiene JSONs válidos aún.")
    exit()

print(f"✅ Se encontraron {len(eval_data)} registros evaluados.")

# Crear DataFrame con las métricas
df_eval = pd.DataFrame([{
    "ID": r.get("ID"),
    "rouge2": r.get("rouge2", 0),
    "rougeL": r.get("rougeL", 0),
    "bleu": r.get("bleu", 0),
    "avg": r.get("avg", 0),
} for r in eval_data])

print("\n📊 Estadísticas globales actuales:")
print(df_eval.describe())

print("\n📌 Promedios generales:")
print(df_eval[["rouge2", "rougeL", "bleu", "avg"]].mean())

