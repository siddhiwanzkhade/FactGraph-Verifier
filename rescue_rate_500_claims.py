import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# ── load files ─────────────────────────────────────────────────
llm_df = pd.read_csv("/Users/siddhiwanzkhade/Downloads/FactGraph/outputs/annotated_500.csv")
nli_df = pd.read_csv("/Users/siddhiwanzkhade/Downloads/FactGraph/evaluation_results_nli.csv")

# ── normalize labels ───────────────────────────────────────────
llm_df["label"]     = llm_df["label"].str.upper().str.strip().str.replace(" ", "_")
llm_df["llm_label"] = llm_df["llm_label"].str.upper().str.strip().str.replace(" ", "_")

nli_df["gold_label"] = nli_df["gold_label"].str.upper().str.strip()
nli_df["pred_label"] = nli_df["pred_label"].str.upper().str.strip()

# ── merge LLM labels with NLI results on claim ─────────────────
merged = llm_df.merge(
    nli_df[["claim", "pred_label", "nli_premise_verbalized", "has_evidence"]],
    on="claim",
    how="inner"
)
merged = merged.drop_duplicates(subset="claim", keep="first")
print(f"Claims with both LLM + NLI labels: {len(merged)}")

# ── rename for clarity ─────────────────────────────────────────
merged = merged.rename(columns={
    "label":     "gold_label",
    "llm_label": "llm_pred",
    "pred_label": "kg_pred"
})

# ── core flags ─────────────────────────────────────────────────
merged["llm_correct"] = merged["llm_pred"]  == merged["gold_label"]
merged["kg_correct"]  = merged["kg_pred"]   == merged["gold_label"]
merged["llm_wrong"]   = ~merged["llm_correct"]
merged["kg_wrong"]    = ~merged["kg_correct"]

# ── hallucination rescue ───────────────────────────────────────
# LLM was wrong AND KG system was right
rescued  = merged[ merged["llm_wrong"] &  merged["kg_correct"]]
# LLM was wrong AND KG system was also wrong
missed   = merged[ merged["llm_wrong"] & ~merged["kg_correct"]]
# LLM was right
llm_hits = merged[ merged["llm_correct"]]

total         = len(merged)
llm_wrong     = merged["llm_wrong"].sum()
rescue_count  = len(rescued)
rescue_rate   = rescue_count / llm_wrong if llm_wrong > 0 else 0

print(f"\n=== ACCURACY COMPARISON ===")
print(f"LLM baseline accuracy:      {merged['llm_correct'].mean():.4f} ({merged['llm_correct'].mean()*100:.2f}%)")
print(f"KG system accuracy:         {merged['kg_correct'].mean():.4f} ({merged['kg_correct'].mean()*100:.2f}%)")

print(f"\n=== HALLUCINATION RESCUE ===")
print(f"Total claims:               {total}")
print(f"LLM wrong:                  {llm_wrong} ({100*llm_wrong/total:.1f}%)")
print(f"Rescued (LLM wrong, KG right): {rescue_count} ({100*rescue_count/total:.1f}%)")
print(f"Hallucination rescue rate:  {rescue_rate:.4f} ({rescue_rate*100:.2f}%)")
print(f"  = {rescue_count} / {llm_wrong} claims where LLM was wrong")

print(f"\n=== WHERE RESCUE HAPPENED ===")
print(rescued.groupby(["gold_label","kg_pred"]).size())

print(f"\n=== RESCUE EXAMPLES ===")
sample = rescued[["claim","gold_label","llm_pred","kg_pred","nli_premise_verbalized"]].head(8)
print(sample.to_string(index=False))

print(f"\n=== CLASSIFICATION REPORTS ===")
print("-- LLM baseline --")
print(classification_report(
    merged["gold_label"], merged["llm_pred"],
    labels=["SUPPORTS","REFUTES","NOT_ENOUGH_INFO"]
))
print("-- KG system --")
print(classification_report(
    merged["gold_label"], merged["kg_pred"],
    labels=["SUPPORTS","REFUTES","NOT_ENOUGH_INFO"]
))

# ── save ───────────────────────────────────────────────────────
merged.to_csv("/Users/siddhiwanzkhade/Downloads/FactGraph/outputs/hallucination_rescue_analysis.csv", index=False)
print(f"Saved → hallucination_rescue_analysis.csv")