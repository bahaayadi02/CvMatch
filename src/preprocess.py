"""Text cleaning and normalisation.

Two cleaning levels:

- `clean_text` is light: lowercases, normalises whitespace, drops obvious
  PDF noise. Used before semantic embedding (Sentence-Transformers
  works better on natural text than aggressively cleaned text).

- `normalize_with_spacy` is heavy: lemmatises, removes stopwords and
  punctuation. Used for the TF-IDF baseline and for the keyword overlap
  feature, where matching "developing" to "develop" matters.
"""
import re

import spacy


_NLP = None


def _get_nlp():
    """Load spaCy lazily so importing this module is cheap."""
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except OSError as e:
            raise OSError(
                "spaCy model 'en_core_web_sm' is not installed. Run:\n"
                "    python -m spacy download en_core_web_sm"
            ) from e
    return _NLP


def clean_text(text):
    """Light cleaning suitable for embedding models."""
    if not text:
        return ""
    # Strip URLs and emails — they confuse the tokenizer and aren't useful
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    # Common PDF artefact: bullet glyphs that didn't decode cleanly
    text = re.sub(r"[•●▪◦‣⁃]", " ", text)
    # Collapse whitespace (PDF extraction often leaves stray newlines)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def normalize_with_spacy(text):
    """Lemmatise and drop stopwords/punctuation. Returns a list of tokens.

    Used for the TF-IDF baseline and for keyword-overlap scoring — both
    benefit from "running" → "run", "developed" → "develop" normalization.
    """
    nlp = _get_nlp()
    doc = nlp(clean_text(text))
    return [
        tok.lemma_
        for tok in doc
        if not tok.is_stop and not tok.is_punct and not tok.is_space and len(tok.lemma_) > 1
    ]


def normalize_to_string(text):
    """Same as normalize_with_spacy but joined into a string for sklearn."""
    return " ".join(normalize_with_spacy(text))
