from fastapi import FastAPI
from google import genai

app = FastAPI()
client = genai.Client(api_key="REPLACE THIS")

@app.get("/ask")
def ask(prompt: str):
    interaction = client.interactions.create(
        model="gemini-3.5-flash",
        input=prompt,
    )
    return {"text": interaction.output_text}