# Architecture

A short tour of how the matcher fits together, with two diagrams.
The first shows the data flow at runtime; the second shows the
file-level dependencies between modules.

## Runtime workflow

```mermaid
flowchart TD
    A[User uploads PDF resume<br/>+ pastes job description] --> B[parser.py<br/>extract text from PDF]
    A --> C[parser.py<br/>read JD text]
    B --> D[preprocess.py<br/>clean + spaCy lemmatise]
    C --> D
    D --> E[skill_extractor.py<br/>dictionary lookup]
    D --> F[matcher.py — semantic similarity<br/>Sentence-Transformers]
    D --> G[matcher.py — TF-IDF similarity<br/>scikit-learn]
    E --> H[matcher.py — skill overlap]
    F --> I[Composite score<br/>0.5 × semantic + 0.2 × tfidf + 0.3 × skill]
    G --> I
    H --> I
    I --> J[recommender.py<br/>rule-based suggestions]
    J --> K[Streamlit UI<br/>app.py]
    H --> K
```

## Module dependency graph

```mermaid
flowchart LR
    parser[parser.py]
    preprocess[preprocess.py]
    skill[skill_extractor.py]
    matcher[matcher.py]
    recommender[recommender.py]
    app[app.py]
    eval[scripts/evaluate.py]

    parser --> matcher
    parser --> eval
    preprocess --> matcher
    preprocess --> skill
    skill --> matcher
    matcher --> recommender
    matcher --> app
    matcher --> eval
    recommender --> app
```

## Why each piece exists

- **parser.py** — input layer. Hides the difference between PDF and text resumes.
- **preprocess.py** — text normalisation. Two cleaning levels: light (for embeddings) and heavy (lemmatised, for TF-IDF and keyword matching).
- **skill_extractor.py** — dictionary-based skill spotter. Transparent and explainable, at the cost of being limited to known terms.
- **matcher.py** — the scoring engine. Three independent signals (semantic, TF-IDF, skill) combined with hand-set weights.
- **recommender.py** — turns the match result into rule-based suggestions. No generation, no LLM.
- **app.py** — Streamlit UI on top of the engine.
- **scripts/evaluate.py** — the experiment: compares the three signals against labelled pairs, produces the comparison table.
