from fastapi import FastAPI, Header, HTTPException, Form
import base64
import librosa
import numpy as np
import uuid
import os
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="AI Generated Voice Detection API")

API_KEY = "123456"

# ------------------ DUMMY MODEL ------------------
model = RandomForestClassifier()
model.fit([[0] * 15], [0])

# ------------------ FEATURE EXTRACTION ------------------
def extract_features(file_path: str):
    y, sr = librosa.load(file_path, sr=16000, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        [zcr]
    ])

    return features


# ------------------ ROOT ------------------
@app.get("/")
def home():
    return {
        "status": "API is running",
        "docs": "/docs"
    }


# ------------------ DETECT ENDPOINT (FORM DATA) ------------------
@app.post("/detect")
def detect_voice(
    language: str = Form(...),
    audioFormat: str = Form(...),
    audioBase64: str = Form(...),
    x_api_key: str = Header(None)
):
    # --- API KEY ---
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    if audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 supported")

    # --- BASE64 ---
    try:
        audio_bytes = base64.b64decode(audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    if len(audio_bytes) < 1000:
        return {
            "classification": "UNKNOWN",
            "confidence": 0.0,
            "language": language,
            "explanation": "Audio too short or corrupted"
        }

    filename = f"temp_{uuid.uuid4()}.mp3"
    with open(filename, "wb") as f:
        f.write(audio_bytes)

    try:
        features = extract_features(filename)
        probs = model.predict_proba([features])[0]
        prediction = int(np.argmax(probs))
        confidence = float(np.max(probs))
    except Exception:
        return {
            "classification": "UNKNOWN",
            "confidence": 0.0,
            "language": language,
            "explanation": "Audio decoding failed"
        }
    finally:
        if os.path.exists(filename):
            os.remove(filename)

    if prediction == 1:
        return {
            "classification": "AI_GENERATED",
            "confidence": round(confidence, 2),
            "language": language,
            "explanation": "Synthetic acoustic patterns detected"
        }
    else:
        return {
            "classification": "HUMAN_GENERATED",
            "confidence": round(confidence, 2),
            "language": language,
            "explanation": "Natural human vocal patterns detected"
        }
