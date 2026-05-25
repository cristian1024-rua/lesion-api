from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import BeitForImageClassification, BeitImageProcessor
from PIL import Image
import torch
import io

app = FastAPI(title="TuDiagnostico API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Cargando modelo...")
model = BeitForImageClassification.from_pretrained(
    "ALM-AHME/beit-large-patch16-224-finetuned-Lesion-Classification-HAM10000-AH-60-20-20"
)
processor = BeitImageProcessor.from_pretrained(
    "ALM-AHME/beit-large-patch16-224-finetuned-Lesion-Classification-HAM10000-AH-60-20-20"
)
print("Modelo cargado exitosamente")

descripciones = {
    "bcc": "Carcinoma basocelular: tipo de cancer de piel de crecimiento lento que aparece como un bulto brillante. Es el mas comun y tratable.",
    "nv": "Nevus melanocitico: lunar comun. Son crecimientos en la piel que varian en color y forma. La mayoria son benignos.",
    "mel": "Melanoma: tipo mas grave de cancer de piel. Se desarrolla a partir de lunares existentes o aparece como una nueva mancha oscura.",
    "akiec": "Queratosis actinica: parches asperos causados por exposicion solar. Pueden ser precursores de cancer de piel.",
    "df": "Dermatofibroma: bulto pequeno, firme y benigno en la piel, generalmente de color marron o rojizo.",
    "vasc": "Lesion vascular: incluye angiomas, hemangiomas y malformaciones vasculares que aparecen como marcas rojas o purpuras.",
    "bkl": "Lentigo solar: manchas de la edad causadas por exposicion solar. Son manchas planas y oscuras, generalmente benignas.",
}

@app.get("/")
def root():
    return {"status": "ok", "message": "TuDiagnostico API funcionando"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]
    
    probs = torch.softmax(logits, dim=-1)
    probabilidad = probs[0][predicted_class_idx].item()
    
    descripcion = descripciones.get(
        predicted_label.lower(),
        "Descripcion no disponible para esta lesion."
    )
    
    return {
        "diagnosis": predicted_label,
        "description": descripcion,
        "probability": round(probabilidad, 4)
    }
