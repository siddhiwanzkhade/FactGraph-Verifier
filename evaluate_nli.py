import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── load files ─────────────────────────────────────────────────
nli_df  = pd.read_csv("/Users/siddhiwanzkhade/Downloads/FactGraph/nli_results_verbalized.csv")
gold_df = pd.read_csv("/Users/siddhiwanzkhade/Downloads/FactGraph/outputs/covered_claims_new.csv")

nli_df  = nli_df.drop_duplicates(subset="claim", keep="first")
gold_df = gold_df.drop_duplicates(subset="claim", keep="first")
merged  = nli_df.merge(gold_df[["claim","label"]], on="claim", how="inner")
print(f"After dedup merge: {len(merged)}")

print(f"NLI results:  {len(nli_df)}")
print(f"Gold labels:  {len(gold_df)}")

# ── merge on claim ─────────────────────────────────────────────
merged = nli_df.merge(gold_df[["claim", "label"]], on="claim", how="inner")
print(f"Merged:       {len(merged)}")

# ── normalize labels ───────────────────────────────────────────
# gold uses: SUPPORTS, REFUTES, NOT ENOUGH INFO
# nli uses:  SUPPORTS, REFUTES, NOT_ENOUGH_INFO
merged["gold_label"] = merged["label"].str.strip().str.replace(" ", "_")
merged["pred_label"] = merged["nli_verdict"].str.strip()

print(f"\nGold label distribution:")
print(merged["gold_label"].value_counts())
print(f"\nPredicted label distribution:")
print(merged["pred_label"].value_counts())

# ── accuracy ───────────────────────────────────────────────────
acc = accuracy_score(merged["gold_label"], merged["pred_label"])
print(f"\n=== ACCURACY: {acc:.4f} ({acc*100:.2f}%) ===")

# ── classification report ──────────────────────────────────────
print(f"\n=== CLASSIFICATION REPORT ===")
print(classification_report(
    merged["gold_label"],
    merged["pred_label"],
    labels=["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"],
    target_names=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
))

# ── confusion matrix ───────────────────────────────────────────
print(f"=== CONFUSION MATRIX ===")
labels = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]
cm = confusion_matrix(merged["gold_label"], merged["pred_label"], labels=labels)
cm_df = pd.DataFrame(cm, index=["True: "+l for l in labels], columns=["Pred: "+l for l in labels])
print(cm_df.to_string())

# ── hallucination rescue rate ──────────────────────────────────
# claims where LLM was wrong AND our system was right
# we don't have LLM labels yet — skip for now
# but let's compute: where did our system correctly catch REFUTES?

print(f"\n=== KEY METRICS ===")
total          = len(merged)
correct        = (merged["gold_label"] == merged["pred_label"]).sum()
supports_correct = ((merged["gold_label"] == "SUPPORTS") & (merged["pred_label"] == "SUPPORTS")).sum()
refutes_correct  = ((merged["gold_label"] == "REFUTES")  & (merged["pred_label"] == "REFUTES")).sum()
nei_correct      = ((merged["gold_label"] == "NOT_ENOUGH_INFO") & (merged["pred_label"] == "NOT_ENOUGH_INFO")).sum()

print(f"Total claims evaluated:     {total}")
print(f"Correct predictions:        {correct} ({100*correct/total:.1f}%)")
print(f"Correct SUPPORTS:           {supports_correct}")
print(f"Correct REFUTES:            {refutes_correct}")
print(f"Correct NOT_ENOUGH_INFO:    {nei_correct}")

# ── error analysis ─────────────────────────────────────────────
print(f"\n=== ERROR ANALYSIS: where did we go wrong? ===")
errors = merged[merged["gold_label"] != merged["pred_label"]]
print(f"Total errors: {len(errors)}")
print(f"\nError breakdown:")
print(errors.groupby(["gold_label", "pred_label"]).size().sort_values(ascending=False))

print(f"\nSample errors (gold=REFUTES, pred=SUPPORTS):")
sample = errors[
    (errors["gold_label"] == "REFUTES") &
    (errors["pred_label"] == "SUPPORTS")
][["claim", "nli_premise_verbalized", "gold_label", "pred_label"]].head(5)
print(sample.to_string(index=False))

# ── save merged for further analysis ──────────────────────────
merged.to_csv("/Users/siddhiwanzkhade/Downloads/FactGraph/evaluation_results_nli.csv", index=False)
print(f"\nSaved → evaluation_results_nli.csv")