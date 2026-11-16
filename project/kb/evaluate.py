"""Score the chunking strategies on a set of test questions.

Each question has the page that answers it and a phrase from the answer. A
retrieved chunk counts as a hit if it's from that page and contains the phrase,
so the same questions work for every strategy even though their chunks differ.

    recall@k  how often a hit is in the top k
    MRR       average of 1 / (position of the first hit), 0 if none in the top k
    tokens    average size of the top k chunks together, what a request would add

Build the indexes first, then run from the project folder:
    uv run python -m kb.embed --strategy all
    uv run python -m kb.evaluate
"""

import argparse

from kb.chunk import CHUNKERS
from kb.load import load_corpus
from kb.retrieve import load_index, retrieve

# (question, page, phrase). Questions are worded like a user would ask, not
# copied from the page, so retrieval has to match meaning and not just words.
CASES = [
    ("How do I make a query parameter optional?", "tutorial-query-params.md", "q: str | None = None"),
    ("How can I force the client to send a certain query parameter?", "tutorial-query-params.md", "needy"),
    ("Will a bool query param accept values like yes or on?", "tutorial-query-params.md", "short=yes"),
    ("How do I limit a path parameter to a fixed list of allowed values?", "tutorial-path-params.md",
     "class ModelName(str, Enum)"),
    ("Can a path parameter hold a file path that has slashes in it?", "tutorial-path-params.md", ":path"),
    ("My /users/me route gets caught by /users/{user_id}, how do I fix it?", "tutorial-path-params.md",
     "/users/me"),
    ("What happens if someone passes text where my path param expects an int?", "tutorial-path-params.md",
     "int_parsing"),
    ("How do I accept JSON data in a POST endpoint?", "tutorial-body.md", "class Item(BaseModel)"),
    ("How do I turn the request body model into a dict?", "tutorial-body.md", "model_dump()"),
    ("How does FastAPI tell body, path and query parameters apart?", "tutorial-body.md",
     "declared to be of the type of a **Pydantic model**"),
    ("My React app on another port can't call my API, what do I set up?", "tutorial-cors.md", "allow_origins"),
    ("Can I allow all origins with * and still send cookies?", "tutorial-cors.md", "allow_credentials"),
    ("What is the OPTIONS request a browser sends before the real one?", "tutorial-cors.md", "preflight"),
    ("How do I reuse the same query parameters across several endpoints?", "tutorial-dependencies.md",
     "Depends(common_parameters)"),
    ("How can I stop repeating the same Annotated Depends type everywhere?", "tutorial-dependencies.md",
     "CommonsDep"),
    ("Should my dependency be async def or a normal def?", "tutorial-dependencies.md", "know what to do"),
    ("How do I send an email after the response has been returned?", "tutorial-background-tasks.md", "add_task"),
    ("What should I use for heavy work that runs outside the web process?", "tutorial-background-tasks.md",
     "Celery"),
    ("How do I write tests for my endpoints?", "tutorial-testing.md", "TestClient(app)"),
    ("Where should the test file go relative to my app?", "tutorial-testing.md", "test_main.py"),
    ("How do I keep the password out of what my endpoint returns?", "tutorial-response-model.md", "UserOut"),
    ("How do I leave out fields that were never set from the response?", "tutorial-response-model.md",
     "response_model_exclude_unset=True"),
    ("How do I turn off the response model for one endpoint?", "tutorial-response-model.md", "response_model=None"),
    ("What command starts the development server?", "tutorial-first-steps.md", "fastapi dev"),
    ("Where do I find the interactive API documentation?", "tutorial-first-steps.md", "/docs"),
    ("Where is the OpenAPI schema for my app?", "tutorial-first-steps.md", "openapi.json"),
    ("How do I install FastAPI?", "tutorial-index.md", '"fastapi[standard]"'),
]


def check_cases(cases=CASES):
    """Fail if a phrase isn't on its page, so a typo can't count as a miss."""
    pages = {page.name: page.text.lower() for page in load_corpus()}
    for question, page, phrase in cases:
        if phrase.lower() not in pages.get(page, ""):
            raise ValueError(f"{phrase!r} is not in {page} (question: {question!r})")


def is_hit(chunk, page, phrase):
    return chunk["page"] == page and phrase.lower() in chunk["text"].lower()


def evaluate(strategy, k=3, cases=CASES):
    """recall@k, MRR, and average tokens in the top k for one strategy, plus the questions it missed."""
    index = load_index(strategy)
    found, reciprocal_ranks, tokens, misses = 0, [], [], []
    for question, page, phrase in cases:
        hits = retrieve(question, k=k, index=index)
        rank = next((i for i, chunk in enumerate(hits, start=1) if is_hit(chunk, page, phrase)), None)
        found += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0)
        tokens.append(sum(chunk["tokens"] for chunk in hits))
        if rank is None:
            misses.append(question)
    return {
        "recall": found / len(cases),
        "mrr": sum(reciprocal_ranks) / len(cases),
        "tokens": sum(tokens) / len(cases),
        "misses": misses,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare chunking strategies on the test questions.")
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument("--misses", action="store_true", help="also list the questions each strategy missed")
    args = parser.parse_args()
    check_cases()
    print(f"{len(CASES)} questions, top {args.k} chunks\n")
    print(f"{'strategy':<9} {'recall@' + str(args.k):>9} {'MRR':>6} {'tokens':>7}")
    for strategy in CHUNKERS:
        result = evaluate(strategy, args.k)
        print(f"{strategy:<9} {result['recall']:>9.0%} {result['mrr']:>6.2f} {result['tokens']:>7.0f}")
        if args.misses:
            for question in result["misses"]:
                print(f"            missed: {question}")
