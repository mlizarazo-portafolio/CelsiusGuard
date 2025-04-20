from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from embedding_generator import EmbeddingGenerator
import torch
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Verificar token de Hugging Face
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    logger.warning("HF_TOKEN no encontrado en variables de entorno. Algunas funcionalidades pueden no estar disponibles.")

try:
    # Cargar modelo PyTorch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Usando dispositivo: {device}")
    
    model_path = "modelo/LSTM_best.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
        
    model = torch.load(model_path, map_location=device)
    model.eval()  # Set the model to evaluation mode
    logger.info("Modelo LSTM cargado exitosamente")

    # Inicializar el generador de embeddings
    generator = EmbeddingGenerator(token=HF_TOKEN)
    logger.info("Generador de embeddings inicializado")
except Exception as e:
    logger.error(f"Error durante la inicialización: {str(e)}")
    raise

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
async def form_post(request: Request, sequence: str = Form(...)):
    try:
        if not sequence:
            raise HTTPException(status_code=400, detail="La secuencia no puede estar vacía")
            
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
        logger.error(f"Error durante la predicción: {str(e)}")
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": f"Error: {str(e)}", 
            "sequence": sequence
        })
