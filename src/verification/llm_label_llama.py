from groq import Groq
import os
import pandas as pd
import time
import json
from sklearn.metrics import classification_report, confusion_matrix


client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("KEY:", os.getenv("GROQ_API_KEY"))

df = pd.read_csv("/Users/siddhiwanzkhade/Downloads/FactGraph/covered_claims_new.csv")
df_sample = df.sample(n=500, random_state=42).copy()

def get_llm_label(claim):
    prompt = f"""
You are a fact verification assistant.
Given a claim, classify it into exactly one of the following labels:
- SUPPORTS
- REFUTES
- NOT ENOUGH INFO

Output ONLY the label.

Claim: {claim}
"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content.strip()

llm_labels = []

for i, claim in enumerate(df_sample["claim"]):
    try:
        label = get_llm_label(claim)
    except Exception as e:
        print(f"Error at row {i}: {e}")
        label = "ERROR"

    llm_labels.append(label)

    if (i + 1) % 50 == 0:
        print(f"{i + 1}/500 done...")

    time.sleep(2)

df_sample["llm_label"] = llm_labels
df_sample["llm_label"] = df_sample["llm_label"].astype(str).str.upper().str.strip()
df_sample["label"] = df_sample["label"].astype(str).str.upper().str.strip()
df_sample["correct"] = df_sample["label"] == df_sample["llm_label"]

valid_df = df_sample[df_sample["llm_label"] != "ERROR"].copy()

accuracy = valid_df["correct"].mean()
print(f"\nBaseline LLM Accuracy: {accuracy:.2%}")
print(f"Hallucination Rate: {1 - accuracy:.2%}")

nei_mask = valid_df["label"] == "NOT ENOUGH INFO"
nei_claims = valid_df[nei_mask]
overconfident = nei_claims[nei_claims["llm_label"] != "NOT ENOUGH INFO"]

overconfidence = 0 if len(nei_claims) == 0 else len(overconfident) / len(nei_claims)
print(f"Overconfidence Rate: {overconfidence:.2%}")
print("(LLM gives confident label when correct answer is NEI)")

print("\nClassification Report:")
print(classification_report(valid_df["label"], valid_df["llm_label"], zero_division=0))

cm = confusion_matrix(
    valid_df["label"],
    valid_df["llm_label"],
    labels=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
)

print("Confusion Matrix:")
print(pd.DataFrame(
    cm,
    index=["true_SUPPORTS", "true_REFUTES", "true_NEI"],
    columns=["pred_SUPPORTS", "pred_REFUTES", "pred_NEI"]
))

df_sample.to_csv(
    "/Users/siddhiwanzkhade/Downloads/FactGraph/annotated_500.csv",
    index=False
)

report = classification_report(
    valid_df["label"],
    valid_df["llm_label"],
    output_dict=True,
    zero_division=0
)

with open("/Users/siddhiwanzkhade/Downloads/FactGraph/baseline_metrics_llama.json", "w") as f:
    json.dump({
        "accuracy": accuracy,
        "hallucination_rate": 1 - accuracy,
        "overconfidence_rate": overconfidence,
        "classification_report": report
    }, f, indent=2)

print("\nSaved annotated_500.csv and baseline_metrics_llama.json")