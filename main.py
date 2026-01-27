from fastapi import FastAPI, Header, HTTPException, Body
from pydantic import BaseModel, Field
import base64
import librosa
import numpy as np
import uuid
import os
from sklearn.ensemble import RandomForestClassifier

# ------------------ APP ------------------
app = FastAPI(title="AI Generated Voice Detection API")

API_KEY = "123456"

# ------------------ REQUEST MODEL ------------------
class VoiceRequest(BaseModel):
    language: str
    audio_format: str = Field(..., alias="audioFormat")
    audio_base64: str = Field(..., alias="audioBase64")

    class Config:
        populate_by_name = True  # accept both snake_case & camelCase


# ------------------ DUMMY / FALLBACK MODEL ------------------
model = RandomForestClassifier()
model.fit([[0] * 15], [0])


# ------------------ FEATURE EXTRACTION ------------------
def extract_features(file_path: str):
    y, sr = librosa.load(file_path, sr=16000, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        [np.mean(spectral_centroid)],
        [np.mean(spectral_rolloff)],
        [np.mean(zcr)]
    ])

    return features


# ------------------ ROOT ENDPOINT ------------------
@app.get("/")
def home():
    return {
        "status": "API is running",
        "docs": "/docs"
    }


# ------------------ DETECTION ENDPOINT ------------------
@app.post("/detect")
def detect_voice(
    request: VoiceRequest = Body(...),
    x_api_key: str = Header(None)
):
    # --- API KEY CHECK ---
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # --- FORMAT CHECK ---
    if request.audio_format.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 audio supported")

    # --- BASE64 DECODE ---
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio data")

    if len(audio_bytes) < 1000:
        return {
            "classification": "UNKNOWN",
            "confidence": 0.0,
            "language": request.language,
            "explanation": "Audio file too short or corrupted"
        }

    # --- SAVE TEMP FILE ---
    filename = f"temp_{uuid.uuid4()}.mp3"
    with open(filename, "wb") as f:
        f.write(audio_bytes)

    # --- PROCESS AUDIO ---
    try:
        features = extract_features(filename)
        probs = model.predict_proba([features])[0]
        prediction = int(np.argmax(probs))
        confidence = float(np.max(probs))
    except Exception:
        return {
            "classification": "UNKNOWN",
            "confidence": 0.0,
            "language": request.language,
            "explanation": "Audio decoding or feature extraction failed"
        }
    finally:
        if os.path.exists(filename):
            os.remove(filename)

    # --- RESPONSE ---
    if prediction == 1:
        return {
            "classification": "AI_GENERATED",
            "confidence": round(confidence, 2),
            "language": request.language,
            "explanation": "Synthetic acoustic patterns detected in the voice sample"
        }
    else:
        return {
            "classification": "HUMAN_GENERATED",
            "confidence": round(confidence, 2),
            "language": request.language,
            "explanation": "Natural vocal variations typical of human speech detected"
        }
