# FactGraph-Verifier
Knowledge Graph Grounded LLM Hallucination Mitigation System

Reducing Hallucinated Labels in LLM Annotation through Knowledge Graph Verification

Overview
FactGraph Annotator is a hybrid symbolic-neural fact verification framework designed to improve the factual reliability of Large Language Model (LLM) generated annotations.
Rather than replacing LLM reasoning, FactGraph acts as an external verification layer by grounding predictions through:
Knowledge Graph retrieval (Neo4j + Wikidata)
Semantic fallback retrieval (Sentence Transformers)
Natural Language Inference (DeBERTa)
Hallucination rescue evaluation
Core Goal
Reduce hallucinated or unsupported LLM predictions by validating claims against structured external evidence.

Key Features 🎯
LLM Baseline Annotation
LLaMA-based annotation pipeline
Generates SUPPORTS / REFUTES / NOT ENOUGH INFO labels
Provides hallucination-prone baseline for evaluation
Knowledge Graph Verification
Neo4j-powered graph database
Wikidata-derived factual knowledge base
Exact symbolic fact verification
Semantic Fallback Retrieval
Sentence Transformer embedding retrieval
Subject-neighborhood semantic search
Improves recall when exact KG matching fails
NLI Verification
DeBERTa-based claim-evidence validation
Produces final grounded predictions
Hallucination Rescue Metrics
Measures how often FactGraph corrects incorrect LLM predictions
Focuses on trustworthiness over raw benchmark accuracy

System Architecture 🏗️
<img width="1920" height="1080" alt="Untitled (Presentation)" src="https://github.com/user-attachments/assets/704bfbc8-18c4-4499-9605-8882aed07748" />



Dataset 
FEVER (Fact Extraction and VERification)
Wikipedia-derived factual claims
Human-verified labels:
SUPPORTS
REFUTES
NOT ENOUGH INFO
Evidence-backed benchmark for fact verification
Why FEVER?
Provides a rigorous benchmark for comparing:
Raw LLM annotation performance
Knowledge Graph grounded verification

Results 📈
Baseline LLM Accuracy
57.14%
KG System Accuracy
45.67%
Hallucination Rescue Rate
59.62%

Key Insight 💡
Although standalone LLMs achieved higher raw classification accuracy, FactGraph corrected:
127 out of 213 incorrect LLM predictions (~60%)
Primary strengths:
Suppresses hallucinations
Detects unsupported claims
Reduces false certainty
Improves factual trustworthiness

Why This Matters
LLM Baseline:
Higher recall
Higher hallucination
Overconfident predictions
FactGraph:
Lower raw accuracy
Higher factual caution
Better uncertainty calibration
External evidence grounding
Bottom line:
FactGraph improves reliability, safety, and factual robustness rather than simply maximizing prediction score.

Installation Guide ⚙️
Quick Setup
git clone https://github.com/yourusername/factgraph-annotator.git
cd factgraph-annotator

conda create -n factgraph python=3.10
conda activate factgraph

pip install -r requirements.txt
python -m spacy download en_core_web_sm

Neo4j Setup
Install Neo4j Desktop
Start local instance
Update credentials in project scripts

Run Full Pipeline ▶️
python fever_wikidata_kg.py
python load_kgfacts_to_neo4j.py
python extract_triple_v8.py
python query_kg.py
python sem_fallback.py
python nli.py
python llm_label_llama.py
python rescue_rate_500_claims.py


Repository Structure 📂
fever_wikidata_kg.py        # KG construction from Wikidata
load_kgfacts_to_neo4j.py   # Load KG into Neo4j
extract_triple_v8.py        # Claim triple extraction
query_kg.py                 # Exact KG verification
sem_fallback.py             # Semantic retrieval fallback
nli.py                      # NLI verification
llm_label_llama.py          # Baseline LLM labeling
rescue_rate_500_claims.py  # Final rescue metrics


Tech Stack 🛠️
Python
spaCy
Neo4j
Wikidata API
Sentence Transformers
Hugging Face Transformers
DeBERTa NLI
LLaMA / Llama-based LLMs
Pandas
Scikit-learn

Technical Challenges Solved 🔍
Entity ambiguity
Property mismatch
Occupation normalization
Nationality normalization
Date normalization
Semantic retrieval precision
Evidence verbalization
Hallucination suppression

Limitations ⚠️
Lower support/refute recall than unconstrained LLMs
KG coverage constraints
Surface-form mismatches
Limited multi-hop reasoning
Conservative NOT ENOUGH INFO bias

Future Work 🚀
Expanded KG property coverage
Better entity disambiguation
Graph-RAG integration
Cross-encoder semantic reranking
Multi-hop graph reasoning
Fine-tuned NLI verification
Confidence calibration

Use Cases 🌍
LLM safety systems
Annotation quality assurance
Enterprise fact-checking
Graph-RAG validation
Hallucination mitigation middleware
AI trustworthiness research

Resume-Ready Summary 📄
Built FactGraph Annotator, a hybrid knowledge graph + semantic retrieval + NLI framework that corrected ~60% of hallucinated LLM predictions on the FEVER benchmark through structured evidence grounding.

Citation 📖
If using FEVER:
Thorne et al., FEVER: a Large-scale Dataset for Fact Extraction and VERification (2018)


Final Contribution 🏁
FactGraph demonstrates that:
Structured external verification can substantially improve LLM factual trustworthiness even when raw standalone classifier accuracy remains lower.
Main achievement:
A practical hallucination mitigation framework for safer, more reliable LLM deployment.

Closing Narrative
FactGraph was built on the belief that powerful AI systems require more than predictive accuracy—they require verifiable trust.
By combining symbolic knowledge graphs with neural reasoning, FactGraph explores how hybrid architectures can make modern LLM systems:
Safer
More explainable
More trustworthy
Better suited for real-world deployment.

