import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIG
# ============================================================

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Siddhiw$7"

INPUT_FILE = "/Users/siddhiwanzkhade/Downloads/FactGraph/extracted_claims_v8.csv"
OUTPUT_FILE = "/Users/siddhiwanzkhade/Downloads/FactGraph/sem_fallback_results_v2.csv"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5

# Retrieval thresholds by fallback scope
THRESHOLDS = {
    "same_subject_same_property": 0.35,
    "same_subject_related_property": 0.38,
    "same_subject_any_property": 0.48
}

GENERIC_RELATIONS = {
    "OCCUPATION",
    "COUNTRY",
    "DATE_OF_BIRTH",
    "INSTANCE_OF",
    "SEX_OR_GENDER"
}

# Related Wikidata-style property groups
RELATED_PROPERTY_GROUPS = {
    "P106": {"OCCUPATION", "POSITION_HELD", "FIELD_OF_WORK"},
    "P27": {"COUNTRY", "COUNTRY_OF_CITIZENSHIP", "PLACE_OF_BIRTH"},
    "P19": {"PLACE_OF_BIRTH", "COUNTRY", "LOCATION"},
    "P20": {"PLACE_OF_DEATH", "COUNTRY", "LOCATION"},
    "P161": {"CAST_MEMBER", "STARRING", "ACTOR"},
    "P57": {"DIRECTOR"},
    "P58": {"SCREENWRITER"},
    "P178": {"DEVELOPER", "MANUFACTURER"},
    "P264": {"RECORD_LABEL"},
    "P577": {"PUBLICATION_DATE", "RELEASE_DATE", "INCEPTION_DATE"},
    "P571": {"INCEPTION_DATE", "PUBLICATION_DATE", "RELEASE_DATE"},
    "P159": {"HEADQUARTERS_LOCATION", "LOCATION", "COUNTRY"},
    "P495": {"COUNTRY_OF_ORIGIN", "COUNTRY"},
}

# Predicate keywords to guess property group if property_id is missing
PREDICATE_HINTS = {
    "release": {"PUBLICATION_DATE", "RELEASE_DATE", "INCEPTION_DATE"},
    "released": {"PUBLICATION_DATE", "RELEASE_DATE", "INCEPTION_DATE"},
    "publish": {"PUBLICATION_DATE", "RELEASE_DATE"},
    "published": {"PUBLICATION_DATE", "RELEASE_DATE"},
    "born": {"PLACE_OF_BIRTH", "DATE_OF_BIRTH", "COUNTRY"},
    "birth": {"PLACE_OF_BIRTH", "DATE_OF_BIRTH"},
    "died": {"PLACE_OF_DEATH", "DATE_OF_DEATH"},
    "death": {"PLACE_OF_DEATH", "DATE_OF_DEATH"},
    "star": {"CAST_MEMBER", "STARRING", "ACTOR"},
    "stars": {"CAST_MEMBER", "STARRING", "ACTOR"},
    "acted": {"CAST_MEMBER", "STARRING", "ACTOR"},
    "directed": {"DIRECTOR"},
    "director": {"DIRECTOR"},
    "wrote": {"SCREENWRITER", "AUTHOR"},
    "written": {"SCREENWRITER", "AUTHOR"},
    "founded": {"FOUNDED_BY", "FOUNDER", "INCEPTION_DATE"},
    "developed": {"DEVELOPER", "MANUFACTURER"},
    "developer": {"DEVELOPER"},
    "label": {"RECORD_LABEL"},
    "occupation": {"OCCUPATION", "POSITION_HELD"},
    "politician": {"OCCUPATION", "POSITION_HELD"},
    "actor": {"OCCUPATION", "CAST_MEMBER"},
    "actress": {"OCCUPATION", "CAST_MEMBER"},
    "country": {"COUNTRY", "COUNTRY_OF_CITIZENSHIP", "COUNTRY_OF_ORIGIN"},
    "nationality": {"COUNTRY_OF_CITIZENSHIP", "COUNTRY"},
    "headquartered": {"HEADQUARTERS_LOCATION"},
    "located": {"LOCATION", "COUNTRY"},
}


# ============================================================
# CONNECTION + MODEL
# ============================================================

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

print("Loading sentence transformer...")
model = SentenceTransformer(MODEL_NAME)


# ============================================================
# HELPERS
# ============================================================

def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).lower().strip()


def normalize_relation(rel):
    return str(rel).upper().strip()


def triple_to_sentence(entity, relation, value):
    rel_phrase = relation.replace("_", " ").lower()
    return f"{entity} {rel_phrase} {value}"


def get_entity_triples(tx, subject):
    result = tx.run("""
        MATCH (e:Entity {name: $subject})-[r]->(v:Value)
        RETURN e.name AS entity, type(r) AS relation, v.name AS value
    """, subject=subject)
    return [(rec["entity"], rec["relation"], rec["value"]) for rec in result]


def fetch_neighborhood(subject):
    with driver.session() as session:
        return session.execute_read(
            get_entity_triples,
            subject.lower().strip()
        )


def infer_related_relations(row):
    """
    Returns a set of relation names that are likely relevant.
    Uses property_id if available; otherwise uses predicate/claim keywords.
    """
    related = set()

    property_id = str(row.get("property_id", "")).strip()
    if property_id in RELATED_PROPERTY_GROUPS:
        related.update(RELATED_PROPERTY_GROUPS[property_id])

    predicate = normalize_text(row.get("predicate", ""))
    claim = normalize_text(row.get("claim", ""))

    for key, rels in PREDICATE_HINTS.items():
        if key in predicate or key in claim:
            related.update(rels)

    return {normalize_relation(r) for r in related}


def relation_matches(relation, target_relations):
    relation = normalize_relation(relation)
    return relation in target_relations


def rank_candidates(claim, candidates, target_relations, scope):
    """
    Rank triples using semantic similarity + property awareness.
    """
    if not candidates:
        return []

    triple_sentences = [
        triple_to_sentence(entity, relation, value)
        for entity, relation, value in candidates
    ]

    claim_vec = model.encode([claim], batch_size=32, show_progress_bar=False)
    triple_vecs = model.encode(triple_sentences, batch_size=32, show_progress_bar=False)

    scores = cosine_similarity(claim_vec, triple_vecs)[0]

    ranked = []

    for score, sent, triple in zip(scores, triple_sentences, candidates):
        entity, relation, value = triple
        relation_norm = normalize_relation(relation)

        final_score = float(score)

        # reward property relevance
        if relation_norm in target_relations:
            final_score += 0.15

        # penalize generic relations when they are not directly relevant
        if relation_norm in GENERIC_RELATIONS and relation_norm not in target_relations:
            final_score -= 0.12

        ranked.append({
            "score_raw": float(score),
            "score_final": final_score,
            "sentence": sent,
            "triple": triple,
            "relation": relation_norm,
            "scope": scope
        })

    ranked = sorted(ranked, key=lambda x: x["score_final"], reverse=True)
    return ranked


def property_aware_fallback(row, triples):
    """
    Tries fallback in stages:
    1. same subject + same/related property
    2. same subject + related property from hints
    3. same subject + any property, but stricter threshold
    """
    claim = str(row["claim"])
    target_relations = infer_related_relations(row)

    # -----------------------------
    # Level 1/2: property-aware candidates
    # -----------------------------
    if target_relations:
        property_candidates = [
            t for t in triples
            if relation_matches(t[1], target_relations)
        ]

        if property_candidates:
            ranked = rank_candidates(
                claim,
                property_candidates,
                target_relations,
                scope="same_subject_related_property"
            )

            threshold = THRESHOLDS["same_subject_related_property"]
            accepted = [r for r in ranked[:TOP_K] if r["score_final"] >= threshold]

            if accepted:
                return accepted, "same_subject_related_property", "matched_property_aware"

    # -----------------------------
    # Level 3: any property fallback
    # -----------------------------
    ranked = rank_candidates(
        claim,
        triples,
        target_relations,
        scope="same_subject_any_property"
    )

    threshold = THRESHOLDS["same_subject_any_property"]
    accepted = [r for r in ranked[:TOP_K] if r["score_final"] >= threshold]

    if accepted:
        return accepted, "same_subject_any_property", "matched_any_property"

    return [], "none", "low_similarity_or_no_relevant_property"


# ============================================================
# MAIN
# ============================================================

df = pd.read_csv(INPUT_FILE)

# Keep your original logic: only run fallback on semantic_search rows
fallback_df = df[df["route"] == "semantic_search"].copy()

print(f"Total rows in input: {len(df)}")
print(f"Rows going to semantic fallback: {len(fallback_df)}")

output = []
total = len(fallback_df)

for i, (_, row) in enumerate(fallback_df.iterrows()):
    if i % 100 == 0:
        print(f"Processing {i}/{total}...")

    subject = row.get("subject", None)

    if not isinstance(subject, str) or subject.strip() == "":
        output.append({
            **row.to_dict(),
            "fallback_used": False,
            "fallback_reason": "missing_subject",
            "fallback_scope": "none",
            "fallback_top_k_facts": None,
            "fallback_top_k_scores": None,
            "fallback_top_k_raw_scores": None,
            "fallback_top_relation_types": None,
            "sem_top_triple": None,
            "sem_top_score": None,
            "sem_evidence": None,
            "target_relations": None
        })
        continue

    triples = fetch_neighborhood(subject)

    if not triples:
        output.append({
            **row.to_dict(),
            "fallback_used": False,
            "fallback_reason": "no_subject_triples_in_kg",
            "fallback_scope": "none",
            "fallback_top_k_facts": None,
            "fallback_top_k_scores": None,
            "fallback_top_k_raw_scores": None,
            "fallback_top_relation_types": None,
            "sem_top_triple": None,
            "sem_top_score": None,
            "sem_evidence": None,
            "target_relations": ",".join(sorted(infer_related_relations(row)))
        })
        continue

    accepted, scope, reason = property_aware_fallback(row, triples)

    if not accepted:
        output.append({
            **row.to_dict(),
            "fallback_used": False,
            "fallback_reason": reason,
            "fallback_scope": scope,
            "fallback_top_k_facts": None,
            "fallback_top_k_scores": None,
            "fallback_top_k_raw_scores": None,
            "fallback_top_relation_types": None,
            "sem_top_triple": None,
            "sem_top_score": None,
            "sem_evidence": None,
            "target_relations": ",".join(sorted(infer_related_relations(row)))
        })
        continue

    top = accepted[0]
    top_triple = top["triple"]

    evidence_sentences = [x["sentence"] for x in accepted]
    evidence_scores = [round(x["score_final"], 4) for x in accepted]
    raw_scores = [round(x["score_raw"], 4) for x in accepted]
    relation_types = [x["relation"] for x in accepted]

    output.append({
        **row.to_dict(),
        "fallback_used": True,
        "fallback_reason": reason,
        "fallback_scope": scope,
        "fallback_top_k_facts": " | ".join(evidence_sentences),
        "fallback_top_k_scores": " | ".join(map(str, evidence_scores)),
        "fallback_top_k_raw_scores": " | ".join(map(str, raw_scores)),
        "fallback_top_relation_types": " | ".join(relation_types),
        "sem_top_triple": f"{top_triple[0]} --[{top_triple[1]}]--> {top_triple[2]}",
        "sem_top_score": round(top["score_final"], 4),
        "sem_evidence": " ".join(evidence_sentences),
        "target_relations": ",".join(sorted(infer_related_relations(row)))
    })

output_df = pd.DataFrame(output)
output_df.to_csv(OUTPUT_FILE, index=False)

print("\nDone!")
print(f"Saved → {OUTPUT_FILE}")
print(f"Total fallback rows: {len(output_df)}")
print(f"Rows with evidence: {output_df['fallback_used'].sum()}")
print(f"Rows without evidence: {(~output_df['fallback_used']).sum()}")

print("\nFallback reason distribution:")
print(output_df["fallback_reason"].value_counts())

print("\nFallback scope distribution:")
print(output_df["fallback_scope"].value_counts())

print("\nTop relation types:")
print(
    output_df["fallback_top_relation_types"]
    .dropna()
    .str.split(" | ")
    .explode()
    .value_counts()
    .head(20)
)

print("\nSample accepted evidence:")
sample = output_df[output_df["fallback_used"] == True][
    ["claim", "target_relations", "fallback_scope", "sem_top_triple", "sem_top_score", "sem_evidence"]
].head(10)

print(sample.to_string(index=False))

driver.close()