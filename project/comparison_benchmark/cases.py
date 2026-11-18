"""Single-turn questions used by the RAG comparison benchmark.

Add a case to CASES to run the same baseline, proxy, exact-cache, and
semantic-cache checks for another question and its paraphrase.
"""

# Only one case for now, as a sanity check. 
# Case 1:
CASES = (
    {
        "id": "interactive-api-docs",
        "question": "Where do I find the interactive API documentation?",
        "paraphrase": "How do I access the interactive API docs?",
        "expected_evidence": "/docs",
        "system": "Answer in one short sentence using only the supplied context.",
        "compression_stages": ["code"],
    },
)
