from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from embedding_generator import EmbeddingGenerator
import joblib
import numpy as np
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Cargar modelo
model = joblib.load("modelo/rf_model_final.pkl")

# Token de Hugging Face (desde variable de entorno)
HF_TOKEN = os.environ.get("HF_TOKEN")
generator = EmbeddingGenerator(token=HF_TOKEN)

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
async def form_post(request: Request, sequence: str = Form(...)):
    try:
        embedding, _ = generator.generate_embeddings([sequence], ["input"])
        prediction = model.predict(np.array(embedding))[0]
        return templates.TemplateResponse("index.html", {"request": request, "result": prediction, "sequence": sequence})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "result": f"Error: {e}", "sequence": sequence})
