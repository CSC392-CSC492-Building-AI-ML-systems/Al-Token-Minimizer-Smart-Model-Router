"""Ask Jev (through OpenRouter's Decisions API) whether a reply gives the expected answer.

Only used for replies that failed their deterministic check, on cases marked
judge=True (see cases.py). Jev gets the question, the expected answer and the
reply.

API reference:
https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request
"""

import httpx

JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.13"
MAX_REPLY_CHARS = 4000  # the answer is near the start; no need to send a whole essay

_INSTRUCTIONS = (
    "Does the response give the same final answer as expected_answer? "
    "Wording, formatting, notation, extra explanation and rounding to a sensible "
    "precision don't matter. A different, missing or hedged final answer does."
)
_CRITERIA = {
    "true": "The response's final answer matches expected_answer.",
    "false": "The response gives a different answer, several conflicting answers, or no answer.",
}


def judge(question, expected, reply, api_key, timeout=60):
    """Return {"probability": P(reply is acceptable), "model": ..., "cost": ...}.

    Raises httpx.HTTPError if the request fails, KeyError if the answer is missing.
    """
    response = httpx.post(
        JEV_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": JEV_MODEL,
            "state": {
                "question": question,
                "expected_answer": expected,
                "response": reply[:MAX_REPLY_CHARS],
            },
            "questions": {
                "matches": {"type": "noul", "instructions": _INSTRUCTIONS, "criteria": _CRITERIA},
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    return {
        "probability": body["answers"]["matches"]["noul"],
        "model": body.get("model"),
        "cost": (body.get("usage") or {}).get("cost"),
    }
