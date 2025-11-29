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

tokenizer = AutoTokenizer.from_pretrained("andrewmos/indian-legal-summaries-finetuned")
model = AutoModelForCausalLM.from_pretrained("andrewmos/indian-legal-summaries-finetuned")
model.eval()
model.to("cuda")   # <- mueve el modelo a GPU
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
jsonl_file = "summaries_con_finetuning.jsonl"
SAVE_EVERY = 5   # <<< GUARDAR CADA 5
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

dataset_eval = dataset_eval.filter(lambda x: (x["id"] not in ids_ok))

print(f"➡️ Total que se procesarán ahora: {len(dataset_eval)}")


# ------------------------------------------------------------
# LOOP PRINCIPAL — GUARDADO CADA 5
# ------------------------------------------------------------
generated_summaries = []
buffer_lines = []  # <<< BUFFER PARA GUARDADO CADA 5

for i, row in enumerate(tqdm(dataset_eval)):

    row_id = row["id"]
    row_input = row["input"]

    # Construcción prompt Alpaca
    prompt = alpaca_prompt.format(INSTRUCTION, row_input, "")

    MAX_CONTEXT = 32768
    MAX_NEW_TOKENS = 2048
    MAX_INPUT_TOKENS = MAX_CONTEXT - MAX_NEW_TOKENS

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    if "### Response:" in decoded:
        prediction = decoded.split("### Response:")[-1].strip()
    else:
        prediction = decoded.strip()

    generated_summaries.append({"ID": row_id, "Summary": prediction})

    # ----------------------------------------------------------
    # AGREGAR AL BUFFER PARA GUARDAR CADA 5
    # ----------------------------------------------------------
    buffer_lines.append(
        json.dumps({"ID": row_id, "Summary": prediction}, ensure_ascii=False) + "\n"
    )

    if (i + 1) % SAVE_EVERY == 0:
        with open(jsonl_file, "a", encoding="utf-8") as f:
            f.writelines(buffer_lines)
        buffer_lines = []


# ------------------------------------------------------------
# GUARDAR LO QUE FALTE DEL BUFFER AL FINAL
# ------------------------------------------------------------
if len(buffer_lines) > 0:
    with open(jsonl_file, "a", encoding="utf-8") as f:
        f.writelines(buffer_lines)


# ------------------------------------------------------------
# REESCRITURA FINAL SIN DUPLICADOS
# ------------------------------------------------------------
print("\n📝 Reescribiendo archivo final sin duplicados...")

for entry in generated_summaries:
    summary_store[entry["ID"]] = entry["Summary"]

with open(jsonl_file, "w", encoding="utf-8") as f:
    for _id, summary in summary_store.items():
        line = json.dumps({"ID": _id, "Summary": summary}, ensure_ascii=False)
        f.write(line + "\n")

print("✅ Archivo limpio reescrito correctamente.")
print(f"📄 Archivo final: {jsonl_file}")