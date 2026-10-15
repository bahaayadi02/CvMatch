"""Generate recommendations for the candidate.

These are simple, rule-based suggestions — no LLM, no generation. The
match result already tells us what's missing; the recommender just
turns it into human-readable advice and prioritises which gaps to
mention first.
"""

# Skills that are usually deal-breakers in tech roles. If the JD asks
# for one of these and the resume doesn't have it, the recommender
# flags it as a high-priority gap.
HIGH_IMPACT_SKILLS = {
    "python", "sql", "machine learning", "deep learning",
    "tensorflow", "pytorch", "aws", "docker", "kubernetes",
    "git", "linux", "natural language processing", "computer vision",
}


def _bucket_missing_skills(missing):
    """Split missing skills into high-impact and nice-to-have buckets."""
    high = [s for s in missing if s in HIGH_IMPACT_SKILLS]
    other = [s for s in missing if s not in HIGH_IMPACT_SKILLS]
    return high, other


def generate_recommendations(match_result):
    """Turn a match-result dict into a list of suggestion strings.

    The structure is intentionally formulaic — rule-based, not generated
    — so the suggestions are predictable and explainable.
    """
    suggestions = []
    score = match_result["match_percentage"]
    missing = match_result["missing_skills"]
    matched = match_result["matched_skills"]
    sem = match_result["semantic_similarity"]
    tfidf = match_result["tfidf_similarity"]

    # 1. Headline judgement based on the composite score.
    if score >= 75:
        suggestions.append(
            f"Strong match ({score}%). The resume aligns well with this role."
        )
    elif score >= 55:
        suggestions.append(
            f"Moderate match ({score}%). A few targeted edits would help."
        )
    else:
        suggestions.append(
            f"Weak match ({score}%). The resume needs significant tailoring "
            f"for this role, or this role may not be the right fit."
        )

    # 2. Missing skills, prioritised.
    if missing:
        high, other = _bucket_missing_skills(missing)
        if high:
            suggestions.append(
                "High-priority skills missing from the resume: "
                + ", ".join(high)
                + ". If you have any experience with these, add it explicitly."
            )
        if other:
            limit = 8  # don't overwhelm the user
            shown = other[:limit]
            extra = f" (and {len(other) - limit} more)" if len(other) > limit else ""
            suggestions.append(
                "Other skills mentioned in the JD but not your resume: "
                + ", ".join(shown) + extra + "."
            )
    else:
        suggestions.append("No skill gaps detected against the JD's keyword list.")

    # 3. Diagnose the gap between semantic and keyword scores.
    # If semantic similarity is high but keyword similarity is low, the
    # resume conveys the right experience but uses different vocabulary —
    # a fixable problem.
    if sem - tfidf > 0.25 and matched:
        suggestions.append(
            "Your resume describes relevant experience but uses different "
            "wording than the job description. Consider rewriting bullet "
            "points using terms from the JD (e.g. " + ", ".join(matched[:3]) + ")."
        )

    # 4. The opposite — keyword match without semantic match — usually
    # means the resume is keyword-stuffed without describing real work.
    if tfidf > 0.5 and sem < 0.4:
        suggestions.append(
            "The resume contains the right keywords but the overall content "
            "doesn't read as closely related to this role. Consider expanding "
            "your bullet points with concrete examples and outcomes."
        )

    # 5. Generic but useful closing tip if the match is weak.
    if score < 55 and matched:
        suggestions.append(
            "Lead with the skills that DO match (" + ", ".join(matched[:5])
            + ") near the top of the resume, where recruiters look first."
        )

    return suggestions
