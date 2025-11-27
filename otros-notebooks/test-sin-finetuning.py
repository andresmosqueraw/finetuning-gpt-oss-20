# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-1b-it-unsloth-bnb-4bit")
model = AutoModelForCausalLM.from_pretrained("unsloth/gemma-3-1b-it-unsloth-bnb-4bit")
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=2040)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import torch
import os
from rouge_score import rouge_scorer
import sacrebleu
from unsloth import FastLanguageModel

# --------------------------------------------
# CONFIG
# --------------------------------------------
jsonl_file = "summaries_sin_finetuning.jsonl"
save_every = 5  # Guardar cada 5 predicciones (aunque solo haya 3 se guarda al final)

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

# --------------------------------------------
# Cargar dataset test
# --------------------------------------------
dataset_eval = load_dataset(
    "andrewmos/indian-legal-summaries-alpaca-format",
    split="test"
)

print(f"Total test samples originales: {len(dataset_eval)}")

# ⭐ SOLO TOMAR LAS 3 PRIMERAS FILAS
dataset_eval = dataset_eval.select(range(3))
print(f"Ejecutando solo con: {len(dataset_eval)} ejemplos (modo prueba)")

# --------------------------------------------
# Preparar modelo
# --------------------------------------------
FastLanguageModel.for_inference(model)
scorer = rouge_scorer.RougeScorer(['rouge2', 'rougeL'], use_stemmer=True)

# Para acumular métricas
all_metrics = []
temp_buffer = []  # <-- para guardar cada 5 antes de escribir

# Borrar archivo JSONL si existe
if os.path.exists(jsonl_file):
    os.remove(jsonl_file)

# --------------------------------------------
# LOOP
# --------------------------------------------
for i, row in enumerate(tqdm(dataset_eval)):

    row_id = row["id"]
    row_input = row["input"]
    row_reference = row["output"]

    # Construcción del prompt alpaca
    prompt = alpaca_prompt.format(INSTRUCTION, row_input, "")

    # Tokenización
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    # Generación
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
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

    # --------------------------------------------
    # MÉTRICAS
    # --------------------------------------------
    rouge = scorer.score(row_reference, prediction)
    rouge2 = rouge["rouge2"].fmeasure
    rougel = rouge["rougeL"].fmeasure

    bleu = sacrebleu.corpus_bleu([prediction], [[row_reference]]).score / 100

    avg = (rouge2 + rougel + bleu) / 3

    all_metrics.append({
        "id": row_id,
        "rouge2": rouge2,
        "rougeL": rougel,
        "bleu": bleu,
        "avg": avg
    })

    # --------------------------------------------
    # GUARDAR EN BUFFER
    # --------------------------------------------
    temp_buffer.append(
        '{"ID": "' + row_id + '", "Summary": "' + prediction.replace('"', "'") + '"}\n'
    )

    # --------------------------------------------
    # GUARDAR CADA save_every
    # (para 3 ejemplos no ejecutará este if)
    # --------------------------------------------
    if (i + 1) % save_every == 0:
        with open(jsonl_file, "a", encoding="utf-8") as f:
            for line in temp_buffer:
                f.write(line)
        temp_buffer = []

# --------------------------------------------
# GUARDAR LO QUE FALTE AL FINAL
# --------------------------------------------
if len(temp_buffer) > 0:
    with open(jsonl_file, "a", encoding="utf-8") as f:
        for line in temp_buffer:
            f.write(line)

# --------------------------------------------
# MÉTRICAS
# --------------------------------------------
df_metrics = pd.DataFrame(all_metrics)

print("\n📊 MÉTRICAS (solo 3 muestras)")
print(df_metrics)
print("\n📄 Archivo generado:", jsonl_file)