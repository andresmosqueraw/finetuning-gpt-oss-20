import json
import os
import pandas as pd
from datasets import load_dataset
from rouge_score import rouge_scorer
import sacrebleu
from tqdm import tqdm

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------
jsonl_file = "summaries_con_finetuning.jsonl"
dataset_name = "andrewmos/indian-legal-summaries-alpaca-format"

# ------------------------------------------------------------
# CARGAR SUMMARIES GENERADOS
# ------------------------------------------------------------
print(f"📂 Cargando summaries generados desde: {jsonl_file}")

summary_store = {}
if os.path.exists(jsonl_file):
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    # Asegurarse de capturar el summary como string limpio
                    summary_store[data.get("ID")] = str(data.get("Summary", "")).strip()
                except Exception as e:
                    print(f"⚠️ Error leyendo línea: {e}")
                    continue
    print(f"   ✔️ {len(summary_store)} summaries cargados.")
else:
    print(f"❌ No se encontró el archivo {jsonl_file}")
    exit()

# ------------------------------------------------------------
# CARGAR DATASET DE REFERENCIA (GROUND TRUTH)
# ------------------------------------------------------------
print(f"📚 Cargando dataset de referencia: {dataset_name} (split='test')")
dataset_eval = load_dataset(dataset_name, split="test")

# ------------------------------------------------------------
# EVALUACIÓN
# ------------------------------------------------------------
scorer = rouge_scorer.RougeScorer(["rouge2", "rougeL"], use_stemmer=True)
all_metrics = []

print("🚀 Calculando métricas...")

# Convertimos dataset a lista para iterar rápido o iteramos directamente
# Solo nos interesan los que tenemos en summary_store
matches = 0

for row in tqdm(dataset_eval):
    row_id = row["id"]
    
    if row_id in summary_store:
        matches += 1
        pred = summary_store[row_id]
        ref = row["output"]

        # Si la predicción está vacía, penaliza o salta. 
        # Aquí calcularemos igual (dará 0 si está vacía).
        
        # ROUGE
        rouge_scores = scorer.score(ref, pred)
        rouge2 = rouge_scores["rouge2"].fmeasure
        rougel = rouge_scores["rougeL"].fmeasure
        
        # BLEU (sacrebleu espera listas de strings)
        # Nota: sacrebleu da score de 0 a 100, lo dividimos por 100 para normalizar a 0-1
        bleu = sacrebleu.corpus_bleu([pred], [[ref]]).score / 100
        
        avg = (rouge2 + rougel + bleu) / 3

        all_metrics.append({
            "id": row_id,
            "rouge2": rouge2,
            "rougeL": rougel,
            "bleu": bleu,
            "avg": avg
        })

# ------------------------------------------------------------
# RESULTADOS
# ------------------------------------------------------------
if matches == 0:
    print("⚠️ No se encontraron coincidencias de IDs entre el archivo JSONL y el dataset de test.")
else:
    df = pd.DataFrame(all_metrics)
    print("\n📊 RESULTADOS DE LA EVALUACIÓN")
    print("------------------------------------------------------------")
    print(df.describe())
    print("------------------------------------------------------------")
    print(f"Promedios Globales ({len(df)} evaluados):")
    print(f"  🔴 ROUGE-2: {df['rouge2'].mean():.4f}")
    print(f"  🔴 ROUGE-L: {df['rougeL'].mean():.4f}")
    print(f"  🔵 BLEU:    {df['bleu'].mean():.4f}")
    print(f"  🟢 PROMEDIO:{df['avg'].mean():.4f}")

