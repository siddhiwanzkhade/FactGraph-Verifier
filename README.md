# FactGraph-Verifier 

### Knowledge-Graph Grounded Annotation Pipeline for LLM Hallucination Detection and Fact Verification

FactGraph-Verifier is a knowledge-graph-grounded verification layer for LLM fact-checking. It takes an LLM's initial SUPPORTS / REFUTES / NOT ENOUGH INFO label, decomposes the claim into a `(subject, property, object)` triple, retrieves matching facts from a Wikidata-derived Neo4j graph, and runs that evidence through an entailment verifier (DeBERTa NLI or Qwen2.5-3B) before letting the original label stand — catching the exact failure mode where a model states a wrong answer with the same confidence as a right one, because nothing in its output exposes what evidence, if any, backs the label.


## Problem 

Ask an LLM to fact-check a claim, and it gives you a label — SUPPORTS, REFUTES, or NOT ENOUGH INFO — with total confidence, whether it's right or wrong. The label comes straight from parametric memory, with nothing behind it to check.

**Claim:** *"Elon Musk was born in Canada."*<br>
**Model says:** `SUPPORTS`<br>
**Actual answer:** Wrong. Born in Pretoria, South Africa.

**FACT** : He lived in Canada. The model mistook that for birthplace.

No evidence. No way to catch it.

### Tested at scale, on FEVER claims:

| Metric | Baseline LLM |
|---|---|
| Accuracy | 57.14% |
| NEI recall | **0.17** |

Faced with a claim it can't verify, the model(LLM) admits "not enough info" only **17%** of the time.

The other 83% — it guesses. And sounds just as sure.


## Solution 🚀
**How verification works:**

1. The claim is parsed into a `(subject, property, object)` triple using spaCy's NER and dependency parsing — reducing the claim to a structured relation.
2. The triple is looked up in Neo4j through three routes — exact match, property-based retrieval, or a semantic fallback using `all-MiniLM-L6-v2` embeddings and cosine similarity, for when the claim is worded differently from how the fact is stored.
3. Retrieved evidence is verbalized into a sentence and passed to a verifier for entailment classification — either `cross-encoder/nli-deberta-v3-small` (NLI) or `Qwen2.5-3B-Instruct` (LLM verifier, strict/relaxed prompting).
4. The verified label is compared against the baseline LLM label to compute hallucination rescue rate.


<table border="2">
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/d4e43235-0660-45c9-b842-7673f46b7fac" alt="FactGraph Pipeline" width="100%">
    </td>
  </tr>
</table>

```
FEVER claim → LLaMA baseline label → triple extraction → KG retrieval → evidence verification → final label
```

The gold label is never used during prediction — only during evaluation.

## Results

### System-Level Accuracy

| System | Accuracy | Key Behavior |
|--------|----------|--------------|
| Baseline LLM | 57.14% | Strong SUPPORTS/REFUTES, blind to missing evidence |
| KG + LLM (Strict) | 46.28% | Conservative, high NEI recall |
| KG + LLM (Relaxed) | 48.29% | Less conservative, more concrete labels |

### Per-Class Recall

| System | SUPPORTS | REFUTES | NEI |
|--------|----------|---------|-----|
| Baseline LLM | 0.79 | 0.82 | 0.17 |
| KG + LLM (Strict) | 0.28 | 0.21 | 0.83 |
| KG + LLM (Relaxed) | 0.38 | 0.48 | 0.57 |


### Hallucination Rescue

```
Baseline LLM wrong on:        213 / 497 evaluation claims
KG verifier corrected:        133 of those 213 errors
Rescue Rate:                  62.44%
```
 🎯 **62.44% of baseline LLM hallucinations were caught and corrected** by the FactGraph-Verifier.

### Knowledge Graph Coverage

| Step | Count |
|------|-------|
| FEVER dev claims loaded | 19,998 |
| Entities appearing 5+ times | 936 |
| Entities matched on Wikidata | 700 |
| Wikidata facts fetched | 3,990 |
| KG-covered FEVER claims | 7,724 (38.6%) |
| Final evaluation subset | 497 annotated claims |

Properties fetched: date of birth, place of birth, country of citizenship, occupation, award received, country of origin, founded by, headquarters.

## Usage 

To set up and run **FactGraph-Verifier**, follow these steps:

1. **Clone the Repository**

```bash
    git clone https://github.com/siddhiwanzkhade/FactGraph-Verifier.git
    cd FactGraph-Verifier
```

2. **Set Up the Environment**

```bash
    conda create -n factgraph python=3.10
    conda activate factgraph
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
```

3. **Configure Neo4j**

    Add your credentials to `.env` and start [Neo4j Desktop](https://neo4j.com/download/):

```bash
    # .env
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=your_password
```

4. **Build the Knowledge Graph**

```bash
    python src/kg_construction/fever_wikidata_kg.py      # extract FEVER entities + fetch Wikidata facts
    python src/kg_construction/load_kgfacts_to_neo4j.py  # load facts into Neo4j
```

5. **Run the Verification Pipeline**

```bash
    python src/verification/llm_label_llama.py       # baseline LLM annotation
    python src/kg_construction/extract_triple_v8.py  # claim → (subject, property, object) triple
    python src/retrieval/query_kg.py                 # exact-match + property-based Neo4j retrieval
    python src/retrieval/sem_fallback.py             # semantic fallback for unmatched claims
    python src/merge/merge.py                        # merge all evidence into verifier input
    python src/verification/nli.py                   # DeBERTa NLI verification
    python src/verification/llm_verify.py            # Qwen LLM verification (strict + relaxed)
    python src/evaluation/evaluate_nli2.py           # evaluate NLI path
    python src/evaluation/evaluate_llm_verify.py     # evaluate LLM path + rescue rate
```

You're now ready to explore and reproduce FactGraph's results!


## Models Supported

| Component | Model | Purpose |
|-----------|-------|---------|
| Baseline annotator | LLaMA 3.1 | Generates initial claim label without retrieval |
| Semantic property mapping | all-MiniLM-L6-v2 | Maps claim relations to Wikidata-style properties |
| Semantic fallback retriever | all-MiniLM-L6-v2 | Cosine-similarity KG neighborhood search |
| NLI verifier | cross-encoder/nli-deberta-v3-small | Entailment / contradiction / neutral classification |
| LLM verifier | Qwen/Qwen2.5-3B-Instruct | Evidence-grounded claim verification (strict + relaxed) |
| KG backend | Neo4j | Stores and queries Wikidata-derived triples |



## Key Design Decisions

- **Why triples, not text retrieval?**
Retrieval systems differ in two ways: *how* they search (the retriever) and *what* they search for (the retrieval unit). FactGraph varies the first, fixes the second.

| | Retriever | Retrieval unit |
|---|---|---|
| FactGraph | Exact match → property lookup → semantic fallback | Fixed — always a KG triple |

Keeping the retrieval unit fixed means every route — however lenient — returns evidence in the same clean, verifiable shape. Precision is highest on exact match (36.3%) and eases off through property lookup (19.0%) and semantic fallback (14.9%), a controlled tradeoff between coverage and precision by design, not a side effect.
- **Why two verifiers?**
DeBERTa NLI is trained on natural-language pairs and often reads semi-structured KG triples as neutral. Qwen handles them more flexibly and catches paraphrases. Strict vs. relaxed prompting tunes the caution-vs-coverage tradeoff. Rescue rate: ~59% (DeBERTa) → 62.44% (Qwen).
- **Why rescue rate, not accuracy?**
57.14% baseline accuracy looks fine until you see NEI recall at 0.17 — it's confidently wrong on 83% of claims it has no evidence for. Rescue rate measures the thing that matters: how often the KG layer catches that.


## Contributing to FactGraph-Verfier 🤝

- **Graph-RAG integration** — replace static Neo4j queries with dynamic graph traversal. Instead of fetching
     pre-stored triples,the retriever walks the KG at inference time, enabling multi-hop reasoning across entity       relationships that single-property   lookup can't reach.
- **Cross-encoder reranking over semantic fallback** — swap cosine-similarity ranking with a cross-encoder that       jointly scores the claim and candidate evidence. Same retrieval architecture,significantly better precision       on ambiguous claims.
- **Confidence-weighted hybrid decisions** — assign a retrieval confidence score per route (exact match > KG          retrieve > semantic fallback) and use it to dynamically weight KG vs LLM predictions rather than hard             overrides. Moves the system toward a learned fusion layer.
- **Fine-tuned NLI on verbalized KG evidence** — DeBERTa performs best on fluent sentence pairs. Fine-tuning on       verbalized KG triples as premises would specialize the verifier for structured evidence without changing the      retrieval pipeline.
- **Scaling to open-domain claims beyond FEVER** — the current KG is FEVER-targeted. Extending to a broader           Wikidata subgraph would make FactGraph applicable to real-world annotation pipelines, social media fact-          checking, and LLM output auditing at scale.


