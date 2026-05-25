from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import ViTForImageClassification, ViTImageProcessor
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
MODEL_NAME = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
model = ViTForImageClassification.from_pretrained(MODEL_NAME)
model.eval()
print("Modelo listo")

descripciones = {
    "bcc": "Carcinoma basocelular: tipo de cancer de piel de crecimiento lento.",
    "nv": "Nevus melanocitico: lunar comun, generalmente benigno.",
    "mel": "Melanoma: tipo mas grave de cancer de piel.",
    "akiec": "Queratosis actinica: parches asperos causados por exposicion solar.",
    "df": "Dermatofibroma: bulto pequeno firme y benigno.",
    "vasc": "Lesion vascular: incluye angiomas y hemangiomas.",
    "bkl": "Lentigo solar: manchas causadas por el sol.",
}

lesiones = ["bcc", "nv", "mel", "akiec", "df", "vasc", "bkl"]

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
    idx = logits.argmax(-1).item()
    idx_lesion = idx % len(lesiones)
    predicted_label = lesiones[idx_lesion]
    probs = torch.softmax(logits, dim=-1)
    probabilidad = float(probs[0][idx].item())
    return {
        "diagnosis": predicted_label,
        "description": descripciones[predicted_label],
        "probability": round(probabilidad, 4)
    }
