# -*- coding: utf-8 -*-
"""
Evaluación del modelo Finetuned Gemma-3 — usando TEST CHUNKED de HuggingFace
    → Comparación justa contra el modelo finetune
    → Versión rápida con batch inference (sin FastModel)
"""

print("📘 test-con-finetuning.py — Evaluación modelo Finetuned en test chunked (batch)")

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

MODEL_FINETUNED  = "andrewmos/gemma-3-finetune-chunked"
HF_DATASET  = "andrewmos/indian-legal-summaries-chat-chunked"

JSONL_OUTPUT = "evaluacion_test_chunked_Finetuned_fast.jsonl"

MAX_NEW_TOKENS = 1024      # 1024 para que vaya más rápido
BATCH_SIZE     = 8         # puedes probar 8 o 12
SAVE_EVERY     = 20        # guarda cada 20 ejemplos

import os

print("CONFIG ✔️")

# ============================================================
# 2. CARGAR MODELO BASE + TOKENIZER (TRANSFORMERS)
# ============================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Cargando modelo base...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_FINETUNED)
model = AutoModelForCausalLM.from_pretrained(MODEL_FINETUNED)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# por si acaso, desactivar gradientes globalmente
torch.set_grad_enabled(False)

print("Modelo Finetuned cargado en:", device)

# ============================================================
# 3. MÉTRICAS
# ============================================================

from rouge_score import rouge_scorer
import sacrebleu
import json
import pandas as pd
from tqdm import tqdm

scorer = rouge_scorer.RougeScorer(["rouge2", "rougeL"], use_stemmer=True)

def compute_scores(pred, ref):
    r = scorer.score(ref, pred)
    rouge2 = r["rouge2"].fmeasure
    rougel = r["rougeL"].fmeasure
    bleu = sacrebleu.corpus_bleu([pred], [[ref]]).score / 100
    avg  = (rouge2 + rougel + bleu) / 3
    return rouge2, rougel, bleu, avg

# ============================================================
# 4. INFERENCIA EN BATCH — MÁS RÁPIDA
# ============================================================

def infer_batch(text_list):
    """
    text_list: lista de strings [BATCH_SIZE] con el input del usuario
    Devuelve lista de predicciones (len(text_list))
    """
    # Tokenización en batch
    encodings = tokenizer(
        text_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8000,
    ).to(device)

    # Generación en batch
    outputs = model.generate(
        **encodings,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        do_sample=False,          # greedy para reproducibilidad
        use_cache=True,
    )

    # Decodificar todas las secuencias
    preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    preds = [p.strip() for p in preds]
    return preds

# ============================================================
# 5. CARGAR TEST SET CHUNKED
# ============================================================

from datasets import load_dataset

print("Cargando test set chunked desde HuggingFace...")
dataset = load_dataset(HF_DATASET, split="test")
print(f"Total ejemplos en TEST = {len(dataset)}")

# ============================================================
# 6. REANUDACIÓN + AUTOSAVE
# ============================================================

evaluados = set()
buffer = []

if os.path.exists(JSONL_OUTPUT):
    print(f"🔄 Cargando resultados previos desde {JSONL_OUTPUT}...")
    with open(JSONL_OUTPUT, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                evaluados.add(data["ID"])
            except:
                pass
    print(f"✔️ {len(evaluados)} ejemplos ya evaluados.")
else:
    print("➡️ No existe archivo previo. Empezando desde cero.")

procesados = 0

# ============================================================
# 7. LOOP EN BATCH SOBRE EL TEST
# ============================================================

batch_ids    = []
batch_inputs = []
batch_refs   = []

for row in tqdm(dataset):

    doc_id = row["ID"]
    if doc_id in evaluados:
        continue

    user_input = row["messages"][0]["content"]  # chunk de juicio
    reference  = row["messages"][1]["content"]  # resumen gold

    batch_ids.append(doc_id)
    batch_inputs.append(user_input)
    batch_refs.append(reference)

    # cuando llegamos al tamaño de batch → inferencia
    if len(batch_inputs) == BATCH_SIZE:
        preds = infer_batch(batch_inputs)

        for ID, pred, ref in zip(batch_ids, preds, batch_refs):
            r2, rl, bleu, avg = compute_scores(pred, ref)
            buffer.append({
                "ID": ID,
                "chunk_pred": pred,
                "reference": ref,
                "rouge2": r2,
                "rougeL": rl,
                "bleu": bleu,
                "avg": avg,
                "chunks": 1,
            })
            evaluados.add(ID)
            procesados += 1

        # vaciar batch
        batch_ids    = []
        batch_inputs = []
        batch_refs   = []

        # autosave periódico
        if procesados % SAVE_EVERY == 0:
            with open(JSONL_OUTPUT, "a", encoding="utf-8") as f:
                for r in buffer:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            buffer = []
            print(f"💾 Guardado parcial → {procesados} ejemplos")

# procesar últimos que no llenaron un batch completo
if batch_inputs:
    preds = infer_batch(batch_inputs)
    for ID, pred, ref in zip(batch_ids, preds, batch_refs):
        r2, rl, bleu, avg = compute_scores(pred, ref)
        buffer.append({
            "ID": ID,
            "chunk_pred": pred,
            "reference": ref,
            "rouge2": r2,
            "rougeL": rl,
            "bleu": bleu,
            "avg": avg,
            "chunks": 1,
        })

# guardado final
with open(JSONL_OUTPUT, "a", encoding="utf-8") as f:
    for r in buffer:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("🎉 Evaluación del modelo Finetuned (batch) COMPLETA.")

# ============================================================
# 8. RESUMEN GLOBAL
# ============================================================

eval_data = []
with open(JSONL_OUTPUT, "r", encoding="utf-8") as f:
    for line in f:
        try:
            eval_data.append(json.loads(line))
        except:
            pass

df_eval = pd.DataFrame([{
    "ID": r["ID"],
    "rouge2": r["rouge2"],
    "rougeL": r["rougeL"],
    "bleu": r["bleu"],
    "avg": r["avg"],
} for r in eval_data])

print("\n📊 Estadísticas globales del modelo Finetuned (batch):")
print(df_eval.describe())

print("\n📌 Promedios generales:")
print(df_eval[["rouge2","rougeL","bleu","avg"]].mean())