# FactGraph-Verifier

### **Reducing Hallucinated Labels in LLM Predictions through Knowledge Graph Verification**

---

## **Overview**
**FactGraph Verifier** is a hybrid symbolic-neural fact verification framework designed to improve the factual reliability of Large Language Model (LLM) generated predictions.

Rather than replacing LLM reasoning, FactGraph acts as an external verification layer by grounding predictions through:

- **Knowledge Graph retrieval (Neo4j + Wikidata)**
- **Semantic fallback retrieval (Sentence Transformers)**
- **Natural Language Inference (DeBERTa)**
- **Hallucination rescue evaluation**

### **Core Goal**
**Reduce hallucinated or unsupported LLM predictions by validating claims against structured external evidence.**

---
<img width="1454" height="1082" alt="38fbea16-99e4-4a30-a637-65e549e07135" src="https://github.com/user-attachments/assets/5779d15b-7325-4e6c-8f93-9ec3627362cd" />

## **Key Features **

### **LLM Baseline Annotation**
- LLaMA-based prediction pipeline
- Generates **SUPPORTS / REFUTES / NOT ENOUGH INFO** labels
- Provides hallucination-prone baseline for evaluation

### **Knowledge Graph Verification**
- Neo4j-powered graph database
- Wikidata-derived factual knowledge base
- Exact symbolic fact verification

### **Semantic Fallback Retrieval**
- Sentence Transformer embedding retrieval
- Subject-neighborhood semantic search
- Improves recall when exact KG matching fails

### **NLI Verification**
- DeBERTa-based claim-evidence validation
- Produces final grounded predictions

### **Hallucination Rescue Metrics**
- Measures how often FactGraph corrects incorrect LLM predictions
- Prioritizes factual trustworthiness over raw benchmark accuracy

---

## **System Architecture **


<img width="1920" height="1080" alt="Untitled (Presentation)" src="https://github.com/user-attachments/assets/704bfbc8-18c4-4499-9605-8882aed07748" />



## **Dataset Overview**

### **FEVER (Fact Extraction and VERification)**
- Wikipedia-derived factual claims
- Human-verified labels:
  - **SUPPORTS**
  - **REFUTES**
  - **NOT ENOUGH INFO**
- Evidence-backed benchmark for fact verification


---

## **Results **

### **Baseline LLM Accuracy**
**57.14%**

### **KG System Accuracy**
**45.67%**

### **Hallucination Rescue Rate**
**59.62%**

---

## **Key Insight **
Although standalone LLMs achieved higher raw classification accuracy, FactGraph corrected:

### **127 out of 213 incorrect LLM predictions (~60%)**

### **Primary strengths:**
- Suppresses hallucinations
- Detects unsupported claims
- Reduces false certainty
- Improves factual trustworthiness

---

## **Why This Matters**

### **LLM Baseline:**
- Higher recall
- Higher hallucination
- Overconfident predictions

### **FactGraph:**
- Lower raw accuracy
- Higher factual caution
- Better uncertainty calibration
- External evidence grounding

### **Bottom line:**
**FactGraph improves reliability, safety, and factual robustness rather than simply maximizing prediction score.**

---

## **Installation Guide **

### **Quick Setup**
```bash
git clone https://github.com/yourusername/factgraph-annotator.git
cd factgraph-annotator

conda create -n factgraph python=3.10
conda activate factgraph

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## **Neo4j Setup **
Install Neo4j Desktop
Start local instance
Update credentials in project scripts



## **Repository Structure **
fever_wikidata_kg.py        # KG construction from Wikidata
load_kgfacts_to_neo4j.py   # Load KG into Neo4j
extract_triple_v8.py        # Claim triple extraction
query_kg.py                 # Exact KG verification
sem_fallback.py             # Semantic retrieval fallback
nli.py                      # NLI verification
llm_label_llama.py          # Baseline LLM labeling
rescue_rate_500_claims.py  # Final rescue metrics


## **Tech Stack **
Python
spaCy
Neo4j
Wikidata API
Sentence Transformers
Hugging Face Transformers
DeBERTa NLI
LLaMA / Llama-based LLMs labels
Pandas
Scikit-learn

## **Technical Challenges Solved **
Entity ambiguity
Property mismatch
Occupation normalization
Nationality normalization
Date normalization
Semantic retrieval precision
Evidence verbalization
Hallucination suppression

## **Limitations **
Lower support/refute recall than unconstrained LLMs
KG coverage constraints
Surface-form mismatches
Limited multi-hop reasoning
Conservative NOT ENOUGH INFO bias

## **Future Work **
Expanded KG property coverage
Better entity disambiguation
Graph-RAG integration
Cross-encoder semantic reranking
Multi-hop graph reasoning
Fine-tuned NLI verification
Confidence calibration


## **Summary  **
FactGraph demonstrates that:
Structured external verification can substantially improve LLM factual trustworthiness even when raw standalone classifier accuracy remains lower.
Main achievement:
A practical hallucination mitigation framework for safer, more reliable LLM deployment.


