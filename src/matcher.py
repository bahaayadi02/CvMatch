"""The matching engine.

Combines three signals into one match score:

1. **Semantic similarity** (cosine of Sentence-Transformer embeddings).
   Captures meaning even when the words differ — e.g. "built REST APIs"
   in the resume should match "API development experience" in the JD.

2. **TF-IDF keyword similarity** (cosine over n-gram TF-IDF vectors).
   Captures the literal-keyword screening that human recruiters and ATS
   systems do. Complements semantic similarity, which can be too forgiving.

3. **Skill overlap ratio** (matched skills / required skills).
   The most interpretable signal: of the skills the JD asks for, what
   fraction are in the resume?

The composite score is a weighted average. The weights are not learned
— they're chosen by hand and documented in `DEFAULT_WEIGHTS`. Tuning
the weights would need a labelled dataset of resume/JD pairs with
"good fit" / "poor fit" judgements; building that is in future work.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import clean_text, normalize_to_string
from .skill_extractor import extract_skills, matched_and_missing


DEFAULT_WEIGHTS = {
    "semantic": 0.5,   # captures meaning beyond exact keywords
    "tfidf": 0.2,      # literal keyword overlap, ATS-style
    "skill": 0.3,      # fraction of required skills present
}


# Embedding model is loaded lazily and cached.
_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2 is small (~90MB), fast on CPU, and the
        # standard choice for sentence similarity with limited resources.
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def semantic_similarity(text_a, text_b):
    """Cosine similarity between Sentence-Transformer embeddings."""
    model = _get_model()
    embeddings = model.encode([clean_text(text_a), clean_text(text_b)])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(np.clip(sim, 0.0, 1.0))


def tfidf_similarity(text_a, text_b):
    """Cosine similarity over TF-IDF vectors (uni- and bi-grams)."""
    a = normalize_to_string(text_a)
    b = normalize_to_string(text_b)
    if not a or not b:
        return 0.0
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    matrix = vectorizer.fit_transform([a, b])
    sim = cosine_similarity(matrix[0], matrix[1])[0][0]
    return float(np.clip(sim, 0.0, 1.0))


def skill_overlap_score(resume_skills, jd_skills):
    """Fraction of JD skills present in the resume. 0 if JD has no skills."""
    if not jd_skills:
        return 0.0
    matched = set(resume_skills) & set(jd_skills)
    return len(matched) / len(jd_skills)


def match(resume_text, jd_text, weights=None, skill_method="hybrid"):
    """Run the full matching pipeline. Returns a dict with all signals,
    the composite score, matched/missing skills, skill method, and weights.
    """
    weights = weights or DEFAULT_WEIGHTS

    resume_skills = extract_skills(resume_text, method=skill_method)
    jd_skills = extract_skills(jd_text, method=skill_method)
    matched, missing = matched_and_missing(resume_skills, jd_skills)

    sem = semantic_similarity(resume_text, jd_text)
    tfidf = tfidf_similarity(resume_text, jd_text)
    skill = skill_overlap_score(resume_skills, jd_skills)

    composite = (
        weights["semantic"] * sem
        + weights["tfidf"] * tfidf
        + weights["skill"] * skill
    )

    return {
        "match_percentage": round(composite * 100, 1),
        "semantic_similarity": round(sem, 3),
        "tfidf_similarity": round(tfidf, 3),
        "skill_overlap": round(skill, 3),
        "matched_skills": matched,
        "missing_skills": missing,
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "skill_method": skill_method,
        "weights": weights,
    }

