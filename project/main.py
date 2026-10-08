import logging
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai

import db
from log_setup import setup_logging

load_dotenv()  # reads GEMINI_API_KEY from project/.env (not committed)
setup_logging()  # writes to project/logs/app.log, see log_setup.py
logger = logging.getLogger(__name__)

app = FastAPI()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set, copy .env.example to .env and add your key")
client = genai.Client(api_key=api_key)


def _ms_since(start):
    return (time.perf_counter() - start) * 1000


# NOTE ON KEEPING TRACK OF STUFF TO WRITE TO THE DB:co to keep track of all the different fields before writing to the sqlite for a
# specific request, use the RequestRecord object (not sure if it's worth having
# this, maybe just a dict is enough?). it's one record per request, save it once
# at the end, even if the request failed:
#   record = db.RequestRecord(endpoint="/ask", model="gemini-3.5-flash")
#   record.input_tokens_raw = 120
#   record.status = db.STATUS_OK 
#   .........
#   db.insert_request(record) <-- this actually saves to the sqlite
@app.get("/ask")
def ask(prompt: str):
    start = time.perf_counter()
    record = db.RequestRecord(endpoint="/ask", model="gemini-3.5-flash")
    try:
        # token counting (CSC-6) goes here, timed the same way into
        # record.latency_count_ms

        logger.info("Prompt sent to gemini-3.5-flash: %r", prompt)
        upstream_start = time.perf_counter()
        try:
            interaction = client.interactions.create(
                model="gemini-3.5-flash",
                input=prompt,
            )
        finally:
            record.latency_upstream_ms = _ms_since(upstream_start)
        return {"text": interaction.output_text}
    finally:
        record.latency_total_ms = _ms_since(start)
