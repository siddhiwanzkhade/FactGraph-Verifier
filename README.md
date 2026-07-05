# FactGraph-Verifier

### Knowledge Graph Grounded Hallucination Reduction for LLM Annotations and Fact Verification

---

## The Problem

LLM-based fact verification typically operates as closed-book classification: given a claim, the model emits a label — SUPPORTS, REFUTES, or NOT ENOUGH INFO — directly from parametric knowledge, with no mechanism exposing what evidence, if any, justifies that label. This decouples confidence from evidential support. The model is equally fluent whether the claim is true, false, or entirely unverifiable from what it learned during training, and nothing in its output distinguishes a grounded answer from a plausible guess.

This is a distinct failure mode from ordinary misclassification. A wrong label produced with high apparent confidence is indistinguishable from a correct one downstream — there is no confidence signal, hedge, or evidence trail an external system can use to flag it. The claim "Elon Musk was born in Canada" illustrates this concretely: the model answers `SUPPORTS`, conflating Musk's time living in Canada with his place of birth (Pretoria, South Africa). The error is systematic, not random — it stems from surface-level association overriding factual grounding.

> **Claim:** "Elon Musk was born in Canada."
> 
> | | Label | Evidence |
> |--|-------|----------|
> | Baseline LLM | `SUPPORTS` ✗ | none |
> | FactGraph KG | `REFUTES` ✓ | Elon Musk → place of birth → Pretoria |

Empirically, this pattern holds at scale, not just in isolated examples. Running a baseline LLM directly on FEVER claims:

| Metric | Baseline LLM |
|--------|-------------|
| Overall accuracy | 57.14% |
| SUPPORTS recall | 0.79 |
| REFUTES recall | 0.82 |
| **NEI recall** | **0.17** |

The NEI recall is the critical figure: the model correctly identifies only **17% of NOT ENOUGH INFO cases**. On the remaining 83%, faced with a claim it has no real evidence for, it commits to a concrete label rather than abstaining. This is precisely the gap FactGraph is designed to close.

We introduce **FactGraph-Verifier**, a verification layer that intercepts the LLM's initial label before it is treated as final and checks it against a Wikidata-derived knowledge graph. Each claim is decomposed into a `(subject, property, object)` triple, matched against the graph through exact, property-based, or semantic retrieval, and the resulting evidence is passed to an entailment verifier (NLI or LLM-based) that either confirms or overrides the original prediction. The goal is not to replace the LLM's reasoning, but to condition its output on external, inspectable evidence rather than accepting it on the model's word alone.

---

## The Solution

The fix isn't to replace the LLM's reasoning — it's to make it accountable. FactGraph adds a structured knowledge graph verification layer that sits between the LLM's initial label and the final answer, independently checking each claim against Wikidata-derived facts and correcting the LLM whenever the evidence disagrees with it.

Getting there wasn't a single clean step.
* Exact-match lookups against the KG kept missing facts that were actually present — a differently formatted date, a claim phrased around a relation the graph didn't recognize outright.
* Rigid retrieval alone wasn't enough, so a semantic fallback was added to catch what exact matching couldn't.
*  Two verifiers — an NLI model and an LLM verifier — were also compared side by side, since neither handled semi-structured KG evidence perfectly on its own.

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

**How verification works:**

1. The claim is parsed into a `(subject, property, object)` triple using spaCy NER and dependency parsing
2. The triple is looked up in Neo4j via three routes — exact match, property-based retrieval, or semantic fallback using `all-MiniLM-L6-v2` cosine similarity
3. Retrieved evidence is verbalized and passed to a verifier — either `cross-encoder/nli-deberta-v3-small` (NLI) or `Qwen2.5-3B-Instruct` (LLM verifier with strict/relaxed prompting)
4. The verified label is compared against the baseline LLM label to measure hallucination rescue

---

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

### Detailed Classification Report

| System | Class | Precision | Recall | F1 |
|--------|-------|-----------|--------|----|
| Baseline LLM | SUPPORTS | 0.62 | 0.79 | 0.69 |
| | REFUTES | 0.52 | 0.82 | 0.64 |
| | NEI | 0.60 | 0.17 | 0.27 |
| KG + LLM (Strict) | SUPPORTS | 0.57 | 0.28 | 0.38 |
| | REFUTES | 0.60 | 0.21 | 0.31 |
| | NEI | 0.42 | 0.83 | 0.56 |
| KG + LLM (Relaxed) | SUPPORTS | 0.58 | 0.38 | 0.46 |
| | REFUTES | 0.51 | 0.48 | 0.50 |
| | NEI | 0.42 | 0.57 | 0.49 |

### Hallucination Rescue

```
Baseline LLM wrong on:        213 / 497 evaluation claims
KG verifier corrected:        133 of those 213 errors
Rescue Rate:                  62.44%
```

The KG verifier trades raw accuracy for hallucination control. It is most valuable as a second-stage correction layer, not a standalone classifier.

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

---

## Folder Structure

```bash
.
├── src/
│   ├── kg_construction/              # Build the knowledge graph
│   │   ├── fever_wikidata_kg.py      # Extract FEVER entities + fetch Wikidata facts
│   │   ├── extract_triple_v8.py      # spaCy NER + dependency parsing → triples
│   │   └── load_kgfacts_to_neo4j.py  # Load Wikidata facts into Neo4j
│   ├── retrieval/                    # KG lookup + semantic fallback
│   │   ├── query_kg.py               # Exact-match and property-based Neo4j retrieval
│   │   └── sem_fallback.py           # Cosine-similarity fallback retrieval
│   ├── merge/                        # Merge all evidence sources
│   │   └── merge.py
│   ├── verification/                 # NLI + LLM verifiers
│   │   ├── llm_label_llama.py        # LLaMA baseline annotation
│   │   ├── nli.py                    # DeBERTa NLI verification
│   │   └── llm_verify.py             # Qwen verifier (strict + relaxed)
│   └── evaluation/                   # Metrics + rescue rate
│       ├── evaluate_nli2.py          # Accuracy + classification report (NLI path)
│       ├── evaluate_llm_verify.py    # Accuracy + classification report (LLM path)
│       └── rescue_rate_500_claims.py # Hallucination rescue rate analysis
├── outputs/                          # Pipeline CSV outputs
├── neo4j_graph_output/               # Neo4j graph snapshots
├── factgraph_pipeline.png
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/siddhiwanzkhade/FactGraph-Verifier.git
cd FactGraph-Verifier
```

### 2. Create Environment

```bash
conda create -n factgraph python=3.10
conda activate factgraph
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Configure Environment Variables

```bash
# .env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### 4. Start Neo4j

Install [Neo4j Desktop](https://neo4j.com/download/), start a local instance, update credentials in `src/kg_construction/load_kgfacts_to_neo4j.py`.

---

## Usage

### Build and Load the Knowledge Graph

```bash
python src/kg_construction/fever_wikidata_kg.py
python src/kg_construction/load_kgfacts_to_neo4j.py
```

### Run the Full Pipeline

```bash
python src/verification/llm_label_llama.py       # baseline LLM annotation
python src/kg_construction/extract_triple_v8.py  # claim → (subject, property, object) triple
python src/retrieval/query_kg.py                 # KG exact-match and property retrieval
python src/retrieval/sem_fallback.py             # semantic fallback for unmatched claims
python src/merge/merge.py                        # merge all evidence into verifier input
python src/verification/nli.py                   # DeBERTa NLI verification
python src/verification/llm_verify.py            # Qwen LLM verification (strict + relaxed)
python src/evaluation/evaluate_nli2.py           # evaluate NLI path
python src/evaluation/evaluate_llm_verify.py     # evaluate LLM path + rescue rate
```

---

## Models

| Component | Model | Purpose |
|-----------|-------|---------|
| Baseline annotator | LLaMA 3.1 | Generates initial claim label without retrieval |
| Semantic property mapping | all-MiniLM-L6-v2 | Maps claim relations to Wikidata-style properties |
| Semantic fallback retriever | all-MiniLM-L6-v2 | Cosine-similarity KG neighborhood search |
| NLI verifier | cross-encoder/nli-deberta-v3-small | Entailment / contradiction / neutral classification |
| LLM verifier | Qwen/Qwen2.5-3B-Instruct | Evidence-grounded claim verification (strict + relaxed) |
| KG backend | Neo4j | Stores and queries Wikidata-derived triples |

---

## Tech Stack

Python · spaCy · Neo4j · Wikidata API · Sentence Transformers · HuggingFace Transformers · DeBERTa · LLaMA · Qwen · Pandas · Scikit-learn

---

## Key Design Decisions

**Why KG triples instead of text retrieval?**

RAG retrieves passages — rich context, but the model still has to locate the relevant fact inside the passage. FactGraph retrieves structured triples like `Anne Rice -- occupation -- novelist` that directly represent the factual relation being checked. Coverage is narrower than text retrieval, but evidence is explicit and inspectable.

**Why two verifiers?**

DeBERTa NLI is trained on natural-language premise–hypothesis pairs and treats semi-structured KG triples as neutral more often than it should. Qwen handles the same triples flexibly and can match paraphrases. The strict vs relaxed prompting controls the caution-vs-coverage trade-off directly. Rescue rate improved from ~59% (DeBERTa) to 62.44% (Qwen).

**Why rescue rate instead of just accuracy?**

Raw accuracy hides class-level failure. The baseline looks strong at 57.14% overall, but NEI recall of 0.17 means it's confidently mislabeling 83% of claims that lack evidence. Rescue rate measures what matters: how often does the verification layer catch those mistakes.

---

## Future Work

1. **Graph-RAG integration** — replace static Neo4j queries with dynamic graph traversal. Instead of fetching
     pre-stored triples,the retriever walks the KG at inference time, enabling multi-hop reasoning across entity       relationships that single-property lookup can't reach.

3. **Cross-encoder reranking over semantic fallback** — swap cosine-similarity ranking with a cross-encoder that       jointly scores the claim and candidate evidence. Same retrieval architecture,significantly better precision       on ambiguous claims.

4. **Confidence-weighted hybrid decisions** — assign a retrieval confidence score per route (exact match > KG          retrieve > semantic fallback) and use it to dynamically weight KG vs LLM predictions rather than hard             overrides. Moves the system toward a learned fusion layer.

5. **Fine-tuned NLI on verbalized KG evidence** — DeBERTa performs best on fluent sentence pairs. Fine-tuning on       verbalized KG triples as premises would specialize the verifier for structured evidence without changing the      retrieval pipeline.

6. **Scaling to open-domain claims beyond FEVER** — the current KG is FEVER-targeted. Extending to a broader           Wikidata subgraph would make FactGraph applicable to real-world annotation pipelines, social media fact-          checking, and LLM output auditing at scale.

---

## Acknowledgments

- [FEVER Dataset](https://fever.ai/dataset/fever.html)
- [Wikidata API](https://www.wikidata.org/wiki/Wikidata:Data_access)
- [Neo4j Documentation](https://neo4j.com/docs/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
