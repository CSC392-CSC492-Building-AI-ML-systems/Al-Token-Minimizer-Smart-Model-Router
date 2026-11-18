"""Single-turn RAG baseline-versus-proxy comparison benchmark.

The package is split by responsibility:
    * setup.py prepares the KB index and Gemini client.
    * cases.py defines the benchmark questions.
    * run.py executes direct, proxy, exact-cache and semantic-cache calls.

Run from the project folder with:
    uv run python -m comparison_benchmark

Each run saves chart-friendly JSON under comparison_benchmark/results/<run>/report.json.
"""

from comparison_benchmark.run import run

run()
