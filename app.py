from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from embedding_generator import EmbeddingGenerator
import torch
import numpy as np
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Cargar modelo PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load("modelo/LSTM_best.pt", map_location=device)
model.eval()  # Set the model to evaluation mode

# Token de Hugging Face (desde variable de entorno)
HF_TOKEN = os.environ.get("HF_TOKEN")
generator = EmbeddingGenerator(token=HF_TOKEN)

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
async def form_post(request: Request, sequence: str = Form(...)):
    try:
        # Generar embeddings
        embedding, _ = generator.generate_embeddings([sequence], ["input"])
        
        # Convertir a tensor de PyTorch
        input_tensor = torch.FloatTensor(np.array(embedding)).to(device)
        
        # Realizar predicción
        with torch.no_grad():
            output = model(input_tensor)
            prediction = output.cpu().numpy()[0]
        
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": prediction, 
            "sequence": sequence
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": f"Error: {str(e)}", 
            "sequence": sequence
        })
