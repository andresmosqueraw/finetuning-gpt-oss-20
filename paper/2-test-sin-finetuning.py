# ------------------------------------------------------------
# 1. IMPORTS
# ------------------------------------------------------------
from typing import List, Dict, Any, Set
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
import json
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from rouge_score import rouge_scorer
import sacrebleu

"""
Script de Evaluación de Modelos de Lenguaje (Sin Fine-Tuning).

Este script carga un modelo de lenguaje pre-entrenado (Gemma-3) en modo 4 bits,
realiza inferencia sobre un dataset de test y calcula métricas de calidad
(ROUGE-2, ROUGE-L y BLEU) comparando los resúmenes generados con los de referencia.

Funcionalidades principales:
1. Carga de modelo optimizada (BitsAndBytes 4-bit).
2. Gestión de estado (resume) para continuar ejecuciones interrumpidas.
3. Inferencia con truncamiento inteligente de contexto.
4. Cálculo y reporte de métricas de evaluación.
"""

# ------------------------------------------------------------
# 2. CONFIGURACIÓN
# ------------------------------------------------------------
HF_DATASET_NAME: str = "andrewmos/indian-legal-summaries-chat-template"
MODEL_NAME: str = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"
MAX_NEW_TOKENS: int = 1024
MAX_INPUT_TOKENS: int = 8000  # Margen de seguridad (Contexto total ~8192)
jsonl_file: str = "summaries_sin_finetuning.jsonl"

# ------------------------------------------------------------
# 3. CARGA DEL MODELO (ESTÁNDAR HUGGING FACE)
# ------------------------------------------------------------
print("🚀 Cargando modelo con Transformers (Modo Estable)...")

# Configuración para cargar en 4 bits (ahorra memoria igual que Unsloth)
bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16, # O bfloat16 si tu GPU es Ampere (A100, A10, 3090...)
    bnb_4bit_quant_type="nf4"
)

try:
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",  # Usa la GPU automáticamente
        torch_dtype=torch.float16,
        attn_implementation="sdpa" # Usa Flash Attention si está disponible (más rápido)
    )
    print("✅ Modelo cargado correctamente en GPU.")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    exit()

# ------------------------------------------------------------
# 4. CARGAR DATASET
# ------------------------------------------------------------
dataset_eval = load_dataset(HF_DATASET_NAME, split="test")
print(f"Total test samples originales: {len(dataset_eval)}")

# ------------------------------------------------------------
# 5. GESTIÓN DE CONTINUACIÓN (RESUME)
# ------------------------------------------------------------
summary_store: Dict[str, str] = {}
if os.path.exists(jsonl_file):
    print(f"📂 Archivo encontrado: {jsonl_file}. Cargando registros previos...")
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data: Dict[str, Any] = json.loads(line)
                    _id: str = data.get("ID", "")
                    summary_store[_id] = str(data.get("Summary", "")).strip()
                except:
                    continue
    print(f"   ✔️ {len(summary_store)} registros cargados")
else:
    print("📝 Archivo nuevo, no hay registros previos.")

# Filtrar ya procesados y vacíos
ids_ok: Set[str] = {k for k, v in summary_store.items() if v and v.lower() not in ["null", "none", ""]}
dataset_eval = dataset_eval.filter(lambda x: (x["ID"] not in ids_ok))
print(f"➡️ Total que se procesarán ahora: {len(dataset_eval)}")

# ------------------------------------------------------------
# 6. LOOP DE INFERENCIA
# ------------------------------------------------------------
generated_summaries: List[Dict[str, str]] = []

print(f"🚀 Iniciando inferencia en {len(dataset_eval)} muestras...")

for i, row in enumerate(tqdm(dataset_eval)):
    row_id: str = row["ID"]
    row_input: str = row["messages"][0]["content"]

    # A. Preparar mensaje
    messages: List[Dict[str, str]] = [{"role": "user", "content": row_input}]

    # B. Tokenizar
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    # C. Truncamiento inteligente
    # Cortamos solo si excede el límite, manteniendo el inicio (instrucción)
    input_len: int = inputs["input_ids"].shape[-1]
    if input_len > MAX_INPUT_TOKENS:
        inputs["input_ids"] = inputs["input_ids"][:, :MAX_INPUT_TOKENS]
        inputs["attention_mask"] = inputs["attention_mask"][:, :MAX_INPUT_TOKENS]

    # D. Generar
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False  # Greedy decoding para evaluación
        )

    # E. Decodificar (cortando el prompt de entrada)
    # Calculamos la longitud real del input usado (puede haber sido truncado)
    len_input_real: int = inputs["input_ids"].shape[-1]
    prediction: str = tokenizer.decode(outputs[0][len_input_real:], skip_special_tokens=True).strip()

    generated_summaries.append({"ID": row_id, "Summary": prediction})

    # Guardado parcial cada 10 items
    if (i + 1) % 10 == 0:
        with open(jsonl_file, "a", encoding="utf-8") as f:
            for item in generated_summaries:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        # Actualizar store y limpiar buffer
        summary_store.update({x["ID"]: x["Summary"] for x in generated_summaries})
        generated_summaries = []

# Guardar remanentes
if generated_summaries:
    with open(jsonl_file, "a", encoding="utf-8") as f:
        for item in generated_summaries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary_store.update({x["ID"]: x["Summary"] for x in generated_summaries})

# ------------------------------------------------------------
# 7. LIMPIEZA FINAL Y MÉTRICAS
# ------------------------------------------------------------
print("\n📝 Reescribiendo archivo final limpio...")
with open(jsonl_file, "w", encoding="utf-8") as f:
    for _id, summary in summary_store.items():
        line = json.dumps({"ID": _id, "Summary": summary}, ensure_ascii=False)
        f.write(line + "\n")

print("\n📊 CALCULANDO MÉTRICAS GLOBALES...")
scorer = rouge_scorer.RougeScorer(["rouge2", "rougeL"], use_stemmer=True)
all_metrics_global: List[Dict[str, Any]] = []

# Recargar dataset completo para comparar
full_dataset = load_dataset(HF_DATASET_NAME, split="test")

for row in full_dataset:
    row_id = row["ID"]
    if row_id in summary_store:
        pred: str = summary_store[row_id]
        ref: str = row["messages"][1]["content"]

        rouge_scores = scorer.score(ref, pred)
        rouge2: float = rouge_scores["rouge2"].fmeasure
        rougel: float = rouge_scores["rougeL"].fmeasure
        
        # Sacrebleu
        bleu: float = sacrebleu.corpus_bleu([pred], [[ref]]).score / 100
        avg: float = (rouge2 + rougel + bleu) / 3

        all_metrics_global.append({
            "id": row_id, "rouge2": rouge2, "rougeL": rougel, "bleu": bleu, "avg": avg
        })

if all_metrics_global:
    df_global = pd.DataFrame(all_metrics_global)
    print(df_global.describe())
    print(f"\nResultados Finales:")
    print(f"  R2: {df_global['rouge2'].mean():.4f} | RL: {df_global['rougeL'].mean():.4f} | BLEU: {df_global['bleu'].mean():.4f}")
    print(f"  AVG: {df_global['avg'].mean():.4f}")
else:
    print("⚠️ No hay métricas disponibles (dataset vacío o errores).")
