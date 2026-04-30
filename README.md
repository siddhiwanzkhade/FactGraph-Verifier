# FactGraph-Verifier

### **Reducing Hallucinated Labels in LLM Predictions through Knowledge Graph Verification**

---

<p align="center">
  <img 
    src="https://github.com/user-attachments/assets/5779d15b-7325-4e6c-8f93-9ec3627362cd" 
    alt="FactGraph KG Verification Example"
    width="750"
  />
</p>

---
## Problem

LLMs can generate confident labels even when the prediction is not supported by evidence.  
This is especially risky in fact verification tasks where the model must decide whether a claim is:

- **SUPPORTED**
- **REFUTED**
- **NOT ENOUGH INFO**

FactGraph-Verifier adds an external verification layer to reduce false certainty and improve factual reliability.

---

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



## **Key Features**
---

1. **LLM Baseline Prediction**  
   A LLaMA-based model predicts the initial claim label.

2. **Triple Extraction**  
   Claims are parsed into subject–predicate–object style representations.

3. **Knowledge Graph Verification**  
   Neo4j is queried for exact evidence from Wikidata-derived facts.

4. **Semantic Fallback Retrieval**  
   If exact KG evidence is missing, Sentence Transformers retrieve semantically similar facts from the subject’s graph neighborhood.

5. **NLI Verification**  
   DeBERTa checks whether the retrieved evidence supports, refutes, or does not provide enough information for the claim.

6. **Hallucination Rescue Evaluation**  
   The system measures how often FactGraph corrects wrong LLM predictions.

---
---

## **System Architecture**


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

## **Results**

### **Baseline LLM Accuracy**
**57.14%**

### **KG System Accuracy**
**45.67%**

### **Hallucination Rescue Rate**
**59.62%**

---
### **Key Result**

FactGraph corrected **127 out of 213 incorrect LLM predictions**, achieving a hallucination rescue rate of approximately **60%**.

This shows that even though the standalone LLM had higher raw accuracy, FactGraph was effective at identifying and correcting many unsupported or hallucinated predictions.

---

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

## **Installation Guide**

### **Quick Setup**
```bash
git clone https://github.com/yourusername/factgraph-annotator.git
cd factgraph-annotator

conda create -n factgraph python=3.10
conda activate factgraph

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## **Neo4j Setup**
Install Neo4j Desktop
Start local instance
Update credentials in project scripts



## **Tech Stack**
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

## **Technical Challenges Solved**
Entity ambiguity
Property mismatch between claims and KG facts
Date and nationality normalization
Occupation and relation normalization
Missing KG evidence
Semantic retrieval precision
Reducing overconfident unsupported predictions

## **Limitations**
The system depends on the coverage of the Knowledge Graph.
Some claims require facts that are missing from the current KG.
Surface-form mismatches can affect retrieval.
Multi-hop reasoning is limited.
The system can be conservative and predict NOT ENOUGH INFO when evidence is incomplete.

## **Future Work**
Expanded KG property coverage
Better entity disambiguation
Graph-RAG integration
Cross-encoder semantic reranking
Multi-hop graph reasoning
Fine-tuned NLI verification
Confidence calibration


## **Summary**
FactGraph-Verifier demonstrates how external knowledge grounding can reduce hallucinated LLM predictions in fact verification tasks.
While the standalone LLM achieved higher raw accuracy, FactGraph corrected nearly 60% of the LLM’s incorrect predictions by validating claims against Knowledge Graph evidence and NLI-based reasoning.
This makes the project useful for building safer and more trustworthy LLM systems where factual reliability matters more than unsupported confidence.


