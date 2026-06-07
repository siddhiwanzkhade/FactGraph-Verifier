# FactGraph-Verifier

Knowledge Graph Grounded Hallucination Correction for LLM Fact Verification

---

## Overview

LLMs label factual claims fast — but without evidence. When the model is wrong, nothing flags it. At annotation scale, confidently wrong labels silently corrupt your dataset.

FactGraph sits between the raw LLM annotation and the final label. It independently verifies each claim against a Wikidata-derived Knowledge Graph, and corrects the LLM when the evidence says otherwise.

Built and evaluated on the [FEVER dataset](https://fever.ai/dataset/fever.html) · CSE 538 NLP · Stony Brook University · May 2026

---

## Pipeline

![FactGraph Pipeline](factgraph_pipeline.png)

```
FEVER claim → LLaMA baseline label → triple extraction → KG retrieval → evidence verification → final label
```

The FEVER gold label is **not used during prediction** — only during evaluation.

---

## Features

**LLM Baseline Annotation**
- LLaMA 3.1 predicts one of `SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO` from the raw claim
- No retrieval, no evidence trail — purely from parametric memory
- Treated as the potentially hallucinated label throughout the pipeline

**Claim Triple Extraction**
- spaCy NER + dependency parsing converts the claim into a `(subject, property_id, object)` triple
- Rule-based mapping normalizes the relation to a Wikidata-style property
- Handles relation direction correction (e.g. person-to-film vs film-to-director) and object normalization

**KG Retrieval — Three Routes**
- **Exact match**: subject + property + object all present → direct Neo4j lookup
- **KG retrieve**: property known but object uncertain → fetch all values for subject + property
- **Semantic fallback**: exact lookup fails → embed claim and subject-neighborhood triples with `all-MiniLM-L6-v2`, rank by cosine similarity

**Evidence Verbalizer**
- Converts structured `(subject, property, value)` triples into short natural-language strings before passing to the verifier
- Example: `anne rice -- occupation -- novelist` → `"Anne Rice occupation novelist"`

**NLI Verification — DeBERTa**
- `cross-encoder/nli-deberta-v3-small` takes KG evidence as premise, FEVER claim as hypothesis
- Maps `entailment → SUPPORTS`, `contradiction → REFUTES`, `neutral → NOT ENOUGH INFO`

**LLM Verification — Qwen**
- `Qwen/Qwen2.5-3B-Instruct` receives the claim + retrieved KG evidence and predicts the label
- **Strict prompt**: predict `NOT ENOUGH INFO` unless evidence clearly and directly supports or refutes
- **Relaxed prompt**: allow semantic equivalence and paraphrases (e.g. "novelist" supports "author")
- Improved rescue rate from ~59% (DeBERTa) to **62.44%** by handling semi-structured KG evidence better

**Hallucination Rescue Evaluation**
- A rescue occurs when the baseline LLM is wrong but the KG verifier is correct
- `Rescue Rate = corrected baseline errors / total baseline errors`
- Final result: **133 / 213 = 62.44%**

---

## Results

### System-Level Accuracy

| System | Accuracy | Key Behavior |
|--------|----------|--------------|
| Baseline LLM | 57.14% | Strong SUPPORTS/REFUTES, weak NEI |
| KG + LLM (Strict) | 46.28% | Conservative, high NEI recall |
| KG + LLM (Relaxed) | 48.29% | Less conservative, more concrete labels |

### Per-Class Recall

| System | SUPPORTS | REFUTES | NEI |
|--------|----------|---------|-----|
| Baseline LLM | 0.79 | 0.82 | 0.17 |
| KG + LLM (Strict) | 0.28 | 0.21 | 0.83 |
| KG + LLM (Relaxed) | 0.38 | 0.48 | 0.57 |

The baseline LLM correctly identified only **17% of NOT ENOUGH INFO** cases — predicting a confident concrete label on the other 83% with no evidence to back it up. That's the failure mode FactGraph targets.

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
Baseline LLM wrong on:     213 / 497 evaluation claims
KG verifier corrected:     133 of those 213 errors
Rescue Rate:               62.44%
```

### Knowledge Graph Coverage

| Step | Count |
|------|-------|
| FEVER dev claims loaded | 19,998 |
| Entities appearing 5+ times | 936 |
| Entities matched on Wikidata | 700 |
| Wikidata facts fetched | 3,990 |
| KG-covered FEVER claims | 7,724 (38.6%) |
| Final evaluation subset | 497 annotated claims |

KG properties fetched: date of birth, place of birth, country of citizenship, occupation, award received, country of origin, founded by, headquarters.

---

## Folder Structure

```bash
.
├── outputs/                        # Pipeline CSV outputs
├── scripts/
│   └── sem_fallback.py             # Semantic fallback retrieval
├── neo4j_graph_output/             # Neo4j graph snapshots
├── factgraph_pipeline.png          # Pipeline diagram
├── fever_wikidata_kg.py            # Extract FEVER entities + fetch Wikidata facts
├── load_kgfacts_to_neo4j.py        # Load facts into Neo4j
├── extract_triple_v8.py            # spaCy NER + dependency parsing → triples
├── query_kg.py                     # KG exact-match and property retrieval
├── merge.py                        # Merge exact-match + KG-retrieve + fallback evidence
├── nli.py                          # DeBERTa NLI verification
├── llm_verify.py                   # Qwen verifier (strict + relaxed prompts)
├── llm_label_llama.py              # LLaMA baseline annotation
├── evaluate_nli2.py                # Accuracy + classification report (NLI path)
├── evaluate_llm_verify.py          # Accuracy + classification report + rescue rate (LLM path)
├── rescue_rate_500_claims.py       # Hallucination rescue analysis
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

Install [Neo4j Desktop](https://neo4j.com/download/), start a local instance, update credentials in `load_kgfacts_to_neo4j.py`.

---

## Usage

### Build and Load the Knowledge Graph

```bash
python fever_wikidata_kg.py          # extract entities + fetch Wikidata facts
python load_kgfacts_to_neo4j.py      # load into Neo4j
```

### Run the Full Pipeline

```bash
python llm_label_llama.py            # baseline LLM annotation
python extract_triple_v8.py          # claim → (subject, property, object) triple
python query_kg.py                   # KG exact-match and property retrieval
python scripts/sem_fallback.py       # semantic fallback for unmatched claims
python merge.py                      # merge all evidence into verifier input
python nli.py                        # DeBERTa NLI verification
python llm_verify.py                 # Qwen LLM verification (strict + relaxed)
python evaluate_nli2.py              # evaluate NLI path
python evaluate_llm_verify.py        # evaluate LLM path + rescue rate
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
RAG retrieves passages like "Anne Rice was an American author" — rich context, but the model still has to locate the relevant fact. FactGraph retrieves structured triples like `Anne Rice -- occupation -- novelist` that directly represent the factual relation being checked. The trade-off is coverage (text > KG) vs interpretability and precision (KG > text).

**Why a two-verifier setup?**
DeBERTa NLI works well on natural-language premise–hypothesis pairs but treats semi-structured KG triples as neutral more often. Qwen handles the same triples flexibly and can match paraphrases (e.g. "novelist" supporting "author"). The strict vs relaxed prompting controls the caution-vs-coverage trade-off directly.

**Why rescue rate instead of just accuracy?**
Raw accuracy hides class-level failure. The baseline LLM looks strong at 57.14%, but it only recognizes 17% of NOT ENOUGH INFO cases — meaning 83% of claims that lack evidence get a confident concrete label anyway. Rescue rate measures what actually matters: how often does the verification layer catch those mistakes.

---

## Limitations

- KG coverage is bounded by Wikidata fact availability — an entity can be present in the graph without the specific property needed to verify a given claim
- Surface-form mismatches (e.g. "Spielberg" vs "Steven Spielberg") cause silent lookup failures
- Strict verifier over-predicts NOT ENOUGH INFO when evidence is partial rather than absent
- Multi-hop reasoning not supported
- Final evaluation subset is 497 claims (KG-overlapping), not full FEVER dev

---

## Future Work

1. **Broader KG property coverage** — fetch more Wikidata properties per entity to improve fact-level hit rate beyond 38.6%
2. **Alias resolution and fuzzy entity linking** — reduce missed retrievals from surface-form mismatches without changing pipeline architecture
3. **Smarter KG override logic** — distinguish partial evidence from missing evidence; defer to LLM label when the KG can't ground the claim, override when a direct match exists
4. **Hybrid decision rules** — use LLM flexibility when evidence is absent, KG grounding when a direct match is available
5. **Graph-RAG integration** — replace static KG queries with graph-based retrieval-augmented generation
6. **Cross-encoder reranking** — replace cosine-similarity fallback with a cross-encoder for better evidence ranking

---

## Contributions

**Siddhi Wanzkhade** — Neo4j loading, triple extraction, property mapping, KG query logic, KG coverage analysis, semantic fallback retrieval (Sentence Transformers), hallucination rescue analysis, exact-match vs fallback evaluation.

**Vidisha Deshpande** — KG construction from FEVER entities and Wikidata, Neo4j loading, KG coverage analysis, semantic fallback, DeBERTa NLI verification, LLM verifier experiments, hallucination rescue analysis.

---

## References

- Thorne et al. (2018). FEVER: A Large-scale Dataset for Fact Extraction and VERification. NAACL-HLT.
- Vrandečić & Krötzsch (2014). Wikidata: A Free Collaborative Knowledgebase. CACM.
- Reimers & Gurevych (2019). Sentence-BERT. EMNLP-IJCNLP.
- He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. ICLR.

---

## Acknowledgments

- [FEVER Dataset](https://fever.ai/dataset/fever.html)
- [Wikidata API](https://www.wikidata.org/wiki/Wikidata:Data_access)
- [Neo4j Documentation](https://neo4j.com/docs/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
