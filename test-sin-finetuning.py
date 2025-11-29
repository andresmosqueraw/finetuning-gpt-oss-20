# ------------------------------------------------------------
# CARGA DEL MODELO
# ------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from rouge_score import rouge_scorer
import sacrebleu

tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-1b-it-unsloth-bnb-4bit")
model = AutoModelForCausalLM.from_pretrained("unsloth/gemma-3-1b-it-unsloth-bnb-4bit")
model.eval()
torch.cuda.empty_cache()

# Test rápido del modelo
messages = [{"role": "user", "content": "Who are you?"}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=2048)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
jsonl_file = "summaries_sin_finetuning.jsonl"
INSTRUCTION = (
    "Provide a concise and accurate summary of the following legal judgment. "
    "Focus on the key facts, the legal reasoning, and the final verdict."
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


# ------------------------------------------------------------
# CARGAR DATASET
# ------------------------------------------------------------
dataset_eval = load_dataset(
    "andrewmos/indian-legal-summaries-alpaca-format",
    split="test"
)

print(f"Total test samples originales: {len(dataset_eval)}")

# ------------------------------------------------------------
# CARGAR JSON EXISTENTE EN MEMORIA COMO DICCIONARIO
# ------------------------------------------------------------
summary_store = {}        # ID → Summary
summaries_nuevos = []     # para guardar los nuevos summaries

if os.path.exists(jsonl_file):
    print(f"📂 Archivo encontrado: {jsonl_file}")
    print(f"Cargando registros previos...")

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    _id = data.get("ID", "")
                    summary_store[_id] = str(data.get("Summary", "")).strip()
                except:
                    continue

    print(f"   ✔️ {len(summary_store)} registros cargados")
else:
    print("📝 Archivo nuevo, no hay registros previos.")


# ------------------------------------------------------------
# IDENTIFICAR IDS BUENOS Y IDS A REPROCESAR
# ------------------------------------------------------------
ids_ok = {k for k, v in summary_store.items() if v not in ["", "null", "none", "\"\"", None]}
ids_malos = {k for k, v in summary_store.items() if v in ["", "null", "none", "\"\"", None]}

print(f"➡️ IDs con summary válido: {len(ids_ok)}")
print(f"➡️ IDs con summary VACÍO: {len(ids_malos)} (se reharán)")

# Filtra dejando:
# - no procesados nunca
# - procesados pero con summary vacío
dataset_eval = dataset_eval.filter(lambda x: (x["id"] not in ids_ok))

print(f"➡️ Total que se procesarán ahora: {len(dataset_eval)}")


# ------------------------------------------------------------
# MÉTRICAS
# ------------------------------------------------------------
scorer = rouge_scorer.RougeScorer(["rouge2", "rougeL"], use_stemmer=True)


# ------------------------------------------------------------
# LOOP PRINCIPAL
# ------------------------------------------------------------
generated_summaries = []   # almacenamos los generados ahora

for i, row in enumerate(tqdm(dataset_eval)):

    row_id = row["id"]
    row_input = row["input"]
    row_reference = row["output"]

    # Construcción prompt Alpaca
    prompt = alpaca_prompt.format(INSTRUCTION, row_input, "")

    # TRUNCACIÓN para evitar OOM
    MAX_CONTEXT = 32768
    MAX_NEW_TOKENS = 2048
    MAX_INPUT_TOKENS = MAX_CONTEXT - MAX_NEW_TOKENS

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to("cuda")

    # GENERACIÓN
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    # Extraer predicción
    if "### Response:" in decoded:
        prediction = decoded.split("### Response:")[-1].strip()
    else:
        prediction = decoded.strip()

    # Guardar para limpieza final
    generated_summaries.append({"ID": row_id, "Summary": prediction})


# ------------------------------------------------------------
# GUARDAR RESULTADOS FINALES SIN DUPLICADOS
# ------------------------------------------------------------
print("\n📝 Reescribiendo archivo final sin duplicados...")

# Actualizar summary_store con los nuevos summaries
for entry in generated_summaries:
    summary_store[entry["ID"]] = entry["Summary"]

# Reescribir archivo completo limpio
with open(jsonl_file, "w", encoding="utf-8") as f:
    for _id, summary in summary_store.items():
        line = json.dumps({"ID": _id, "Summary": summary}, ensure_ascii=False)
        f.write(line + "\n")

print("✅ Archivo limpio reescrito correctamente.")


print(f"\n📄 Archivo final: {jsonl_file}")
print(f"   Registros finales: {len(summary_store)}")

# ------------------------------------------------------------
# MÉTRICAS GLOBALES SOBRE TODOS LOS SUMMARIES (archivo completo)
# ------------------------------------------------------------
print("\n📊 MÉTRICAS GLOBALES SOBRE TODOS LOS SUMMARIES ALMACENADOS")

all_metrics_global = []

# Recorrer el dataset completo
full_dataset = load_dataset("andrewmos/indian-legal-summaries-alpaca-format", split="test")

for row in full_dataset:
    row_id = row["id"]

    # Solo evaluar si ese ID existe en summary_store
    if row_id in summary_store:
        pred = summary_store[row_id]
        ref = row["output"]

        # Calcular ROUGE/ BLEU
        rouge_scores = scorer.score(ref, pred)
        rouge2 = rouge_scores["rouge2"].fmeasure
        rougel = rouge_scores["rougeL"].fmeasure
        bleu = sacrebleu.corpus_bleu([pred], [[ref]]).score / 100

        avg = (rouge2 + rougel + bleu) / 3

        all_metrics_global.append({
            "id": row_id,
            "rouge2": rouge2,
            "rougeL": rougel,
            "bleu": bleu,
            "avg": avg
        })

df_global = pd.DataFrame(all_metrics_global)

print(df_global.describe())
print(f"\nPromedios globales:")
print(f"  ROUGE-2: {df_global['rouge2'].mean():.4f}")
print(f"  ROUGE-L: {df_global['rougeL'].mean():.4f}")
print(f"  BLEU: {df_global['bleu'].mean():.4f}")
print(f"  Promedio: {df_global['avg'].mean():.4f}")
print(f"\n📌 Evaluados: {len(df_global)} summaries en total")