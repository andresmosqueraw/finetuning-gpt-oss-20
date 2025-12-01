from transformers import AutoTokenizer
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-1b-it-unsloth-bnb-4bit")

df = pd.read_json("./just-nlp-folders/datasets/train/train_judg.jsonl", lines=True)
df_summary = pd.read_json("./just-nlp-folders/datasets/train/train_ref_summ.jsonl", lines=True)

token_lens = df["Judgment"].apply(lambda x: len(tokenizer(x)["input_ids"]))
token_lens_summary = df_summary["Summary"].apply(lambda x: len(tokenizer(x)["input_ids"]))

print("Promedio tokens:", token_lens.mean())
print("Máximo tokens:", token_lens.max())
print("Promedio tokens summary:", token_lens_summary.mean())
print("Máximo tokens summary:", token_lens_summary.max())