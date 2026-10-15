"""Skill extraction from resumes and job descriptions.

Supports two extraction paradigms:
1. **JobBERT Neural NER** (fine-tuned on the SkillSpan benchmark):
   Extracts technical and hard skills dynamically from context, discovering novel
   technologies and tools even if they are not in a pre-compiled list.
2. **Dictionary Matching**:
   Fast, transparent regex matching with alias and acronym normalization
   (e.g., 'k8s' -> 'kubernetes', 'ml' -> 'machine learning').
3. **Hybrid Mode (Default)**:
   Unifies JobBERT NER with dictionary matching for maximum recall and accuracy.
"""
import logging
import re
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)

# Canonical alias mapping for acronyms, variations, and abbreviations
SKILL_ALIASES = {
    "k8s": "kubernetes",
    "ml": "machine learning",
    "nlp": "natural language processing",
    "dl": "deep learning",
    "cv": "computer vision",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "gcp": "google cloud",
    "rest apis": "rest api",
    "restful api": "rest api",
    "restful apis": "rest api",
    "api development": "rest api",
    "neural network": "neural networks",
    "transformer": "transformers",
    "llm": "large language models",
    "llms": "large language models",
    "large language model": "large language models",
    "react.js": "react",
    "reactjs": "react",
    "node": "node.js",
    "nodejs": "node.js",
    "vue.js": "vue",
    "vuejs": "vue",
    "tf": "tensorflow",
    "postgres": "postgresql",
    "gh actions": "github actions",
}

# Curated skill dictionary
SKILL_DICTIONARY = {
    # --- Programming languages ---
    "python", "java", "c++", "c#", "javascript", "typescript", "go",
    "rust", "ruby", "php", "scala", "kotlin", "swift", "matlab", "r",
    "sql", "bash", "shell scripting", "html", "css",

    # --- Web / frontend ---
    "react", "angular", "vue", "node.js", "express",
    "next.js", "tailwind",

    # --- Backend / APIs ---
    "django", "flask", "fastapi", "spring boot", "rest api", "graphql",
    "microservices",

    # --- Data / ML / AI ---
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "reinforcement learning", "neural networks",
    "transformers", "large language models", "data analysis",
    "data visualization", "statistics", "linear algebra", "calculus",
    "feature engineering", "model deployment", "mlops",
    "pandas", "numpy", "scipy", "scikit-learn", "tensorflow", "pytorch",
    "keras", "xgboost", "lightgbm", "spacy", "nltk", "huggingface",
    "sentence transformers", "opencv", "matplotlib", "seaborn", "plotly",
    "statsmodels", "time-series",

    # --- Databases ---
    "postgresql", "mysql", "mongodb", "redis", "sqlite", "elasticsearch",
    "cassandra", "neo4j", "snowflake", "bigquery",

    # --- Cloud / DevOps ---
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "terraform", "ansible", "ci/cd", "jenkins", "github actions",
    "gitlab ci", "linux", "git",

    # --- Tools / methodology ---
    "agile", "scrum", "jira", "confluence", "unit testing",
    "test driven development", "code review",

    # --- ECE-relevant ---
    "embedded systems", "vhdl", "verilog", "fpga", "microcontrollers",
    "arduino", "raspberry pi", "signal processing", "control systems",
    "pcb design", "analog electronics", "digital electronics",
    "iot", "communication systems",

    # --- Soft / general ---
    "communication", "teamwork", "problem solving", "leadership",
    "project management",
}

# Union of canonical skills and aliases for dictionary matching
ALL_DICT_ENTRIES = set(SKILL_DICTIONARY) | set(SKILL_ALIASES.keys())


def canonicalize_skill(skill: str) -> str:
    """Normalize skill variations to their canonical representation."""
    s = skill.strip().lower()
    return SKILL_ALIASES.get(s, s)


def _build_pattern():
    """Compile regex matching any dictionary skill or alias as whole words."""
    sorted_skills = sorted(ALL_DICT_ENTRIES, key=len, reverse=True)
    escaped = [re.escape(s) for s in sorted_skills]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)


_PATTERN = _build_pattern()

# Lazy-loaded JobBERT NER pipeline
_NER_PIPELINE = None


def _get_ner_pipeline():
    """Lazily load the JobBERT skill extraction pipeline."""
    global _NER_PIPELINE
    if _NER_PIPELINE is None:
        try:
            from transformers import pipeline
            # jjzha/jobbert_knowledge_extraction is trained on the SkillSpan dataset
            # specifically for extracting technical knowledge and hard skills.
            _NER_PIPELINE = pipeline(
                "token-classification",
                model="jjzha/jobbert_knowledge_extraction",
                aggregation_strategy="first",
            )
        except Exception as e:
            logger.warning("Could not initialize JobBERT NER pipeline: %s", e)
            return None
    return _NER_PIPELINE


def extract_skills_dictionary(text: str) -> List[str]:
    """Extract skills via dictionary lookup with alias normalization."""
    if not text:
        return []
    found: Set[str] = set()
    for m in _PATTERN.finditer(text):
        canon = canonicalize_skill(m.group(0))
        found.add(canon)
    return sorted(found)


def extract_skills_ner(text: str) -> List[str]:
    """Extract skills using the deep-learning JobBERT token classification model."""
    if not text:
        return []
    nlp = _get_ner_pipeline()
    if nlp is None:
        logger.warning("JobBERT pipeline unavailable, falling back to dictionary.")
        return extract_skills_dictionary(text)

    # Split into lines to respect BERT's 512 token context window limit
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    extracted: Set[str] = set()

    for line in lines:
        if len(line) < 3:
            continue
        try:
            entities = nlp(line)
        except Exception:
            continue

        current_span = None
        current_end = -1
        for ent in entities:
            tag = ent.get("entity_group", "")
            start = ent.get("start", 0)
            end = ent.get("end", 0)

            # Reconstruct contiguous B and I spans
            if tag == "B":
                if current_span:
                    extracted.add(current_span)
                current_span = line[start:end].strip()
                current_end = end
            elif tag == "I":
                if current_span is not None and (start - current_end <= 2):
                    current_span = line[line.find(current_span):end].strip()
                    current_end = end
                elif current_span is not None:
                    current_span += " " + line[start:end].strip()
                    current_end = end
                else:
                    current_span = line[start:end].strip()
                    current_end = end

        if current_span:
            extracted.add(current_span)

    # Post-process spans: strip punctuation, split occasional comma-glued phrases, canonicalize
    clean_skills: Set[str] = set()
    for s in extracted:
        # If model captured "git, linux", split them
        parts = re.split(r"[,;/]+", s)
        for part in parts:
            item = part.strip(" ,.:;()[]{}*•-").lower()
            if len(item) > 1 and not item.isnumeric():
                clean_skills.add(canonicalize_skill(item))

    return sorted(clean_skills)


def extract_skills(text: str, method: str = "hybrid") -> List[str]:
    """Extract skills using 'hybrid' (JobBERT + dictionary), 'ner', or 'dictionary'.

    Returns a sorted list of canonical skill names.
    """
    if not text:
        return []

    method = (method or "hybrid").lower()
    if method == "ner":
        return extract_skills_ner(text)
    elif method == "dictionary":
        return extract_skills_dictionary(text)
    else:  # hybrid
        dict_skills = set(extract_skills_dictionary(text))
        try:
            ner_skills = set(extract_skills_ner(text))
            return sorted(dict_skills | ner_skills)
        except Exception as e:
            logger.warning("Error in NER extraction, returning dictionary results: %s", e)
            return sorted(dict_skills)


def matched_and_missing(resume_skills: List[str], jd_skills: List[str]) -> Tuple[List[str], List[str]]:
    """Compare two skill lists using canonicalized keys.

    Returns (matched, missing):
        matched — skills the JD asks for that the resume has
        missing — skills the JD asks for that the resume does NOT have
    """
    resume_set = {canonicalize_skill(s) for s in resume_skills}
    jd_set = {canonicalize_skill(s) for s in jd_skills}

    matched = sorted(jd_set & resume_set)
    missing = sorted(jd_set - resume_set)
    return matched, missing
