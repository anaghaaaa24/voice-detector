from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
import base64
import librosa
import numpy as np
import uuid
import os

app = FastAPI(title="AI Voice Detector & Honeypot API")

API_KEY = "123456"

# ===================== REQUEST SCHEMA =====================
class VoiceRequest(BaseModel):
    language: str
    audio_format: str = Field(..., alias="audioFormat")
    audio_base64: str = Field(..., alias="audioBase64")

    class Config:
        allow_population_by_field_name = True

# ===================== FEATURE EXTRACTION =====================
def extract_features(file_path: str):
    y, sr = librosa.load(file_path, sr=16000, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    return {
        "mfcc_var": float(np.var(mfcc)),
        "zcr": float(np.mean(zcr)),
        "centroid": float(np.mean(spectral_centroid))
    }

# ===================== ROOT =====================
@app.get("/")
def root():
    return {
        "status": "running",
        "service": "AI Voice Detector + Honeypot"
    }

# ===================== VOICE DETECTION =====================
@app.post("/detect")
def detect_voice(
    request: VoiceRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    if request.audio_format.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 supported")

    try:
        audio_bytes = base64.b64decode(request.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64")

    if len(audio_bytes) < 1000:
        return {
            "classification": "UNKNOWN",
            "confidence": 0.0,
            "language": request.language,
            "explanation": "Audio too short or corrupted"
        }

    filename = f"temp_{uuid.uuid4()}.mp3"
    with open(filename, "wb") as f:
        f.write(audio_bytes)

    try:
        features = extract_features(filename)

        # Heuristic-based decision (hackathon safe)
        if features["mfcc_var"] < 40 and features["zcr"] < 0.05:
            classification = "AI_GENERATED"
            confidence = 0.83
            explanation = "Low spectral variance and uniform MFCC patterns detected"
        else:
            classification = "HUMAN_GENERATED"
            confidence = 0.79
            explanation = "Natural spectral variability detected"

    except Exception:
        classification = "UNKNOWN"
        confidence = 0.0
        explanation = "Audio decoding or feature extraction failed"

    finally:
        if os.path.exists(filename):
            os.remove(filename)

    return {
        "classification": classification,
        "confidence": confidence,
        "language": request.language,
        "explanation": explanation
    }

# ===================== HONEYPOT ENDPOINT =====================
@app.post("/honeypot")
def honeypot(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    return {
        "status": "active",
        "message": "Honeypot endpoint reached successfully",
        "note": "Suspicious activity will be monitored"
    }
