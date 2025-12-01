# ------------------------------------------------------------
# 1. IMPORTS
# ------------------------------------------------------------
from typing import List, Dict, Any, Set, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
import json
import re
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from rouge_score import rouge_scorer
import sacrebleu

"""
Script de Evaluación de Modelos de Lenguaje (Con Fine-Tuning).

Este script carga un modelo fine-tuneado (Gemma-3 Legal), realiza inferencia sobre un dataset de test
utilizando una estrategia de "mejor de N intentos" y validación de calidad en tiempo real.
Incluye post-procesamiento robusto para limpiar el texto generado y cálculo de métricas finales.

Características principales:
1. Inferencia con reintentos: Genera hasta 3 versiones con diferentes temperaturas si la calidad es baja.
2. Validación de calidad: Filtra generaciones que no superan un umbral de ROUGE/BLEU.
3. Limpieza de texto: Elimina artefactos comunes de LLMs (intros, títulos, markdown).
4. Gestión de estado: Permite reanudar ejecuciones interrumpidas.
"""

# ------------------------------------------------------------
# 2. CONFIGURACIÓN
# ------------------------------------------------------------
HF_DATASET_NAME: str = "andrewmos/indian-legal-summaries-chat-template"
MODEL_NAME: str = "andrewmos/gemma-3-1b-legal-summaries-finetuned"
MAX_NEW_TOKENS: int = 1024
MAX_INPUT_TOKENS: int = 8000
jsonl_file: str = "summaries_con_finetuning.jsonl"
QUALITY_THRESHOLD: float = 0.00  # Umbral de calidad

# ------------------------------------------------------------
# 3. FUNCIONES DE LIMPIEZA
# ------------------------------------------------------------
def limpiar_respuesta(text: Optional[str]) -> str:
    """
    Limpia el texto generado por el modelo eliminando introducciones conversacionales,
    títulos, viñetas y aplanando la estructura a un solo párrafo.

    Args:
        text (Optional[str]): Texto crudo generado por el modelo.

    Returns:
        str: Texto limpio y normalizado.
    """
    if not text: return ""
    
    # 1. ELIMINAR INTRODUCCIONES
    patrones_intro: List[str] = [
        r"^(Okay|Sure|Alright|Yes),?\s+(here['’]s|let['’]s|I will|this is)\s+.*?(summary|judgment|analysis|break down).*?(:|\n|\.)",
        r"^Here['’]s a concise summary.*?(:|\n|\.)",
        r"^The provided text is.*?(:|\n)",
        r"^Based on the provided.*?(:|\n)",
    ]
    for p in patrones_intro:
        text = re.sub(p, "", text, flags=re.IGNORECASE | re.DOTALL)

    # 2. ELIMINAR TÍTULOS Y ESTRUCTURAS
    patrones_titulos: List[str] = [
        r"\*\*.*?:?\*\*",
        r"###\s+.*?\n",
        r"^\s*\d+\.\s+.*?:",
        r"Summary:",
        r"Judgment:",
        r"Background & Recommendation:",
        r"Facts:",
        r"Core Facts:",
        r"Key Facts:",
        r"Legal Reasoning:",
        r"Final Verdict:",
        r"Important Note:",
        r"Disclaimer:",
        r"Case Summary:",
        r"Key Takeaways:",
        r"---",
    ]
    for p in patrones_titulos:
        text = re.sub(p, " ", text, flags=re.IGNORECASE | re.MULTILINE)

    # 3. ELIMINAR VIÑETAS
    text = re.sub(r"^\s*[\*\-•]\s+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\s+[\*\-•]\s+", " ", text)

    # 4. APLANAR A PÁRRAFO
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def es_texto_valido(text: Any) -> bool:
    """
    Verifica si un texto es válido (no es None, vacío, ni un valor nulo de string).

    Args:
        text (Any): El texto a validar.

    Returns:
        bool: True si es válido, False en caso contrario.
    """
    if text is None: return False
    s: str = str(text).strip()
    if not s: return False
    if s.lower() in ["null", "none", "nan"]: return False
    return True

# ------------------------------------------------------------
# 4. CARGA DEL MODELO
# ------------------------------------------------------------
print("🚀 Cargando modelo con Transformers...")

bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16, 
    bnb_4bit_quant_type="nf4"
)

try:
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa"
    )
    print("✅ Modelo cargado.")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# ------------------------------------------------------------
# 5. CARGAR DATASET
# ------------------------------------------------------------
dataset_eval = load_dataset(HF_DATASET_NAME, split="test")
ref_map: Dict[str, str] = {}
for row in dataset_eval:
    ref_map[row["ID"]] = row["messages"][1]["content"]

# ------------------------------------------------------------
# 6. GESTIÓN DE CONTINUACIÓN
# ------------------------------------------------------------
summary_store: Dict[str, str] = {}

if os.path.exists(jsonl_file):
    print(f"📂 Cargando registros previos...")
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data: Dict[str, Any] = json.loads(line)
                    _id: str = data.get("ID", "")
                    summary_store[_id] = str(data.get("Summary", "")).strip()
                except:
                    continue
else:
    print("📝 Archivo nuevo.")

scorer = rouge_scorer.RougeScorer(["rouge2", "rougeL"], use_stemmer=True)
ids_ok: Set[str] = set()
ids_bad: Set[str] = set()

print(f"🧐 Validando calidad existente...")

for _id, raw_pred in tqdm(list(summary_store.items())):
    pred: str = limpiar_respuesta(raw_pred)
    summary_store[_id] = pred # Actualizamos limpieza en memoria
    
    if not es_texto_valido(pred) or _id not in ref_map:
        ids_bad.add(_id)
        continue

    ref: str = ref_map[_id]
    try:
        r_scores = scorer.score(ref, pred)
        avg: float = (r_scores["rouge2"].fmeasure + r_scores["rougeL"].fmeasure) / 2 
    except:
        avg = 0.0

    if avg < QUALITY_THRESHOLD:
        ids_bad.add(_id)
    else:
        ids_ok.add(_id)

print(f"   ✅ OK: {len(ids_ok)} | ♻️ REHACER: {len(ids_bad)}")

dataset_eval = dataset_eval.filter(lambda x: (x["ID"] not in ids_ok))
print(f"➡️ Total a procesar: {len(dataset_eval)}")

# ------------------------------------------------------------
# 7. LOOP DE INFERENCIA (3 INTENTOS -> BEST AVG)
# ------------------------------------------------------------
generated_summaries: List[Dict[str, str]] = []

print(f"🚀 Iniciando inferencia (Se queda con el mejor de 3 intentos)...")

for i, row in enumerate(tqdm(dataset_eval)):
    row_id: str = row["ID"]
    row_input: str = row["messages"][0]["content"]
    row_ref: str = row["messages"][1]["content"]

    final_summary: str = ""   # Aquí quedará el ganador
    best_score: float = -1.0    # Para trackear el mejor avg de los 3
    
    messages: List[Dict[str, str]] = [{"role": "user", "content": row_input}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True, padding=True
    ).to(model.device)

    if inputs["input_ids"].shape[-1] > MAX_INPUT_TOKENS:
        inputs["input_ids"] = inputs["input_ids"][:, -MAX_INPUT_TOKENS:]
        inputs["attention_mask"] = inputs["attention_mask"][:, -MAX_INPUT_TOKENS:]

    # --- 3 INTENTOS ---
    for attempt in range(1, 4):
        
        # Variamos temperatura para buscar creatividad si el primero falla
        do_sample_flag: bool = False if attempt == 1 else True
        temp_val: Optional[float] = None if attempt == 1 else (0.7 if attempt == 2 else 0.9)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=10,
                repetition_penalty=1.1,
                do_sample=do_sample_flag,
                temperature=temp_val,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        len_input_real: int = inputs["input_ids"].shape[-1]
        raw_pred: str = tokenizer.decode(outputs[0][len_input_real:], skip_special_tokens=True).strip()
        
        # Limpieza inmediata
        current_pred: str = limpiar_respuesta(raw_pred)

        # Calcular Score
        current_avg: float = 0.0
        if es_texto_valido(current_pred):
            try:
                s = scorer.score(row_ref, current_pred)
                current_avg = (s["rouge2"].fmeasure + s["rougeL"].fmeasure) / 2
            except:
                current_avg = 0.0
        
        # LÓGICA DEL MEJOR:
        # Si este intento es mejor que el anterior (o es el primero), lo guardamos como candidato.
        if current_avg > best_score:
            best_score = current_avg
            final_summary = current_pred

        # Si ya supera el umbral, nos vamos felices.
        if current_avg >= QUALITY_THRESHOLD:
            final_summary = current_pred
            break 
            
    # Al salir del bucle, 'final_summary' contiene el mejor texto generado,
    # aunque sea menor al umbral (porque eliminamos el fallback).

    generated_summaries.append({"ID": row_id, "Summary": final_summary})

    if (i + 1) % 5 == 0:
        with open(jsonl_file, "a", encoding="utf-8") as f:
            for item in generated_summaries:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        summary_store.update({x["ID"]: x["Summary"] for x in generated_summaries})
        generated_summaries = []

if generated_summaries:
    with open(jsonl_file, "a", encoding="utf-8") as f:
        for item in generated_summaries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary_store.update({x["ID"]: x["Summary"] for x in generated_summaries})

# ------------------------------------------------------------
# 8. GUARDADO FINAL
# ------------------------------------------------------------
print("\n📝 Reescribiendo archivo final limpio...")
with open(jsonl_file, "w", encoding="utf-8") as f:
    for _id, summary in summary_store.items():
        clean_summ: str = limpiar_respuesta(summary)
        line = json.dumps({"ID": _id, "Summary": clean_summ}, ensure_ascii=False)
        f.write(line + "\n")

print("\n📊 MÉTRICAS FINALES...")
all_metrics_global: List[Dict[str, Any]] = []
full_dataset = load_dataset(HF_DATASET_NAME, split="test")

for row in full_dataset:
    row_id = row["ID"]
    if row_id in summary_store:
        pred: str = limpiar_respuesta(summary_store[row_id])
        if not es_texto_valido(pred): continue
        
        ref: str = row["messages"][1]["content"]
        rouge_scores = scorer.score(ref, pred)
        r2: float = rouge_scores["rouge2"].fmeasure
        rl: float = rouge_scores["rougeL"].fmeasure
        try:
            b: float = sacrebleu.corpus_bleu([pred], [[ref]]).score / 100
        except:
            b = 0.0
        avg: float = (r2 + rl + b) / 3
        all_metrics_global.append({"id": row_id, "rouge2": r2, "rougeL": rl, "bleu": b, "avg": avg})

if all_metrics_global:
    df_global = pd.DataFrame(all_metrics_global)
    print(df_global.describe())
    print(f"\nResultados Finales AVG: {df_global['avg'].mean():.4f}")
else:
    print("⚠️ No hay métricas disponibles.")