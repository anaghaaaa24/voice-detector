from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import librosa
import numpy as np
import uuid
import os

app = FastAPI(title="AI Voice Detector API")

# ================= CONFIG =================
API_KEY = "123456"

# ================= REQUEST MODEL =================
class VoiceRequest(BaseModel):
    language: str
    audio_format: str
    audio_base64: str

# ================= FEATURE EXTRACTION =================
def extract_features(file_path: str):
    y, sr = librosa.load(file_path, sr=16000, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    features = np.array([
        np.mean(mfcc),
        np.var(mfcc),
        np.mean(zcr),
        np.mean(spectral_centroid)
    ])

    return features

# ================= ROOT ENDPOINT =================
@app.get("/")
def root():
    return {"status": "API is running"}

# ================= DETECT ENDPOINT =================
@app.post("/detect")
def detect_voice(
    request: VoiceRequest,
    x_api_key: str = Header(None)
):
    # 🔐 API KEY CHECK
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 🎵 FORMAT CHECK
    if request.audio_format.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 audio supported")

    # 🔓 BASE64 DECODE
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

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

        # 🧠 SIMPLE HEURISTIC (Hackathon-safe)
        if features[1] < 40 and features[2] < 0.05:
            classification = "AI_GENERATED"
            confidence = 0.82
            explanation = "Low spectral variance and uniform MFCC patterns detected"
        else:
            classification = "HUMAN_GENERATED"
            confidence = 0.78
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
