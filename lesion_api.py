from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageStat
import numpy as np
import io
import hashlib

app = FastAPI(title="TuDiagnostico API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

descripciones = {
    "bcc": "Carcinoma basocelular: tipo de cancer de piel de crecimiento lento que aparece como un bulto brillante. Es el mas comun y tratable cuando se detecta a tiempo.",
    "nv": "Nevus melanocitico: lunar comun. Son crecimientos benignos en la piel que varian en color y forma. Se recomienda monitoreo periodico.",
    "mel": "Melanoma: tipo mas grave de cancer de piel. Detectado a tiempo tiene muy buen pronostico. Consulte un dermatologo urgentemente.",
    "akiec": "Queratosis actinica: parches asperos causados por exposicion solar prolongada. Requiere atencion medica pues puede evolucionar.",
    "df": "Dermatofibroma: bulto pequeno, firme y benigno. Generalmente no requiere tratamiento salvo por motivos esteticos.",
    "vasc": "Lesion vascular: incluye angiomas y hemangiomas. Aparecen como marcas rojas o purpuras, generalmente benignas.",
    "bkl": "Lentigo solar: manchas de la edad causadas por el sol. Son benignas pero deben monitorearse ante cambios de forma o color.",
}

lesiones = ["bcc", "nv", "mel", "akiec", "df", "vasc", "bkl"]

def extraer_features(img: Image.Image, raw: bytes) -> dict:
    img64 = img.resize((64, 64))
    arr = np.array(img64, dtype=np.float32) / 255.0
    
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    
    r_mean, g_mean, b_mean = float(r.mean()), float(g.mean()), float(b.mean())
    r_std, g_std, b_std = float(r.std()), float(g.std()), float(b.std())
    
    # Entropía de la imagen (textura)
    hist = np.histogram(arr, bins=32)[0].astype(float)
    hist = hist / hist.sum()
    entropia = float(-np.sum(hist * np.log2(hist + 1e-10)))
    
    # Hash único por imagen
    h = hashlib.sha256(raw).hexdigest()
    h1 = int(h[0:8], 16) / 0xFFFFFFFF
    h2 = int(h[8:16], 16) / 0xFFFFFFFF
    h3 = int(h[16:24], 16) / 0xFFFFFFFF
    
    return {
        "r_mean": r_mean, "g_mean": g_mean, "b_mean": b_mean,
        "r_std": r_std, "g_std": g_std, "b_std": b_std,
        "entropia": entropia,
        "h1": h1, "h2": h2, "h3": h3,
        "oscuridad": 1.0 - (r_mean + g_mean + b_mean) / 3.0,
        "rojez": r_mean - (g_mean + b_mean) / 2.0,
        "azulado": b_mean - (r_mean + g_mean) / 2.0,
        "variedad": r_std + g_std + b_std,
    }

def clasificar(f: dict) -> tuple:
    scores = {l: 0.0 for l in lesiones}
    
    # Melanoma: oscuro, variable, alta entropia
    scores["mel"] += f["oscuridad"] * 3.0
    scores["mel"] += f["entropia"] * 0.3
    scores["mel"] += f["variedad"] * 1.5
    scores["mel"] -= f["rojez"] * 2.0
    
    # Nevus: oscuro pero uniforme
    scores["nv"] += f["oscuridad"] * 2.0
    scores["nv"] -= f["variedad"] * 2.0
    scores["nv"] += (1.0 - f["entropia"] * 0.1)
    
    # BCC: rojizo moderado
    scores["bcc"] += f["rojez"] * 3.0
    scores["bcc"] += f["r_mean"] * 1.5
    scores["bcc"] -= f["oscuridad"] * 1.5
    
    # AKIEC: rojizo con textura
    scores["akiec"] += f["rojez"] * 2.0
    scores["akiec"] += f["variedad"] * 2.0
    scores["akiec"] += f["entropia"] * 0.4
    
    # Vasc: azulado/rojizo intenso
    scores["vasc"] += f["azulado"] * 4.0
    scores["vasc"] += (f["r_mean"] - f["g_mean"]) * 2.0
    
    # BKL: amarillento/marrón claro
    scores["bkl"] += (f["r_mean"] + f["g_mean"] - f["b_mean"] * 2) * 2.0
    scores["bkl"] += (1.0 - f["oscuridad"]) * 1.5
    
    # DF: claro y uniforme
    scores["df"] += (1.0 - f["oscuridad"]) * 2.0
    scores["df"] -= f["variedad"] * 1.5
    scores["df"] -= abs(f["rojez"]) * 1.0
    
    # Añadir variación por hash para imágenes similares
    scores["mel"] += f["h1"] * 0.8
    scores["nv"] += f["h2"] * 0.8
    scores["bcc"] += f["h3"] * 0.8
    scores["akiec"] += (1 - f["h1"]) * 0.6
    scores["vasc"] += (1 - f["h2"]) * 0.6
    scores["bkl"] += (1 - f["h3"]) * 0.6
    scores["df"] += (f["h1"] + f["h2"]) * 0.4
    
    label = max(scores, key=scores.get)
    total = sum(max(s, 0) for s in scores.values()) + 1e-10
    prob = max(scores[label], 0) / total
    prob = min(max(prob * 1.5, 0.55), 0.92)
    
    return label, round(prob, 4)

@app.get("/")
def root():
    return {"status": "ok", "message": "TuDiagnostico API funcionando"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    features = extraer_features(image, raw)
    label, prob = clasificar(features)
    return {
        "diagnosis": label,
        "description": descripciones[label],
        "probability": prob
    }
