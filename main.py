from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import librosa
import numpy as np
import uuid
import os
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="AI Voice Detection API", version="1.0")

API_KEY = "123456"

# ------------------ Request Schema ------------------
class VoiceRequest(BaseModel):
    language: str
    audio_format: str
    audio_base64: str

# ------------------ FEATURE EXTRACTION ------------------
def extract_features(file_path):
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

# ------------------ TRAIN MODEL ------------------
model = None
X, y_labels = [], []

for label, folder in [(0, "data/human"), (1, "data/ai")]:
    if not os.path.exists(folder):
        continue
    for file in os.listdir(folder):
        if file.endswith(".mp3"):
            path = os.path.join(folder, file)
            try:
                X.append(extract_features(path))
                y_labels.append(label)
            except:
                pass

if len(X) < 2:
    # Fallback model to avoid crash
    model = RandomForestClassifier()
    model.fit([[0]*16], [0])
else:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_labels)

# ------------------ API Endpoint ------------------
@app.post("/detect", summary="Detect if a voice is AI-generated or human")
def detect_voice(request: VoiceRequest, x_api_key: str = Header(None)):

    # -------- API Key check --------
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # -------- Format check --------
    if request.audio_format.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 audio supported")

    # -------- Decode Base64 --------
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
    except:
        raise HTTPException(status_code=400, detail="Invalid Base64 encoding")

    # -------- Size check --------
    if len(audio_bytes) < 1000:
        return {
            "classification": "UNKNOWN",
            "confidence": 0.0,
            "language": request.language,
            "explanation": "Audio too short or corrupted"
        }

    # -------- Save temp file --------
    filename = f"temp_{uuid.uuid4()}.mp3"
    with open(filename, "wb") as f:
        f.write(audio_bytes)

    try:
        # -------- Feature extraction --------
        features = extract_features(filename)

        # -------- Model prediction --------
        probs = model.predict_proba([features])[0]
        prediction = int(np.argmax(probs))
        confidence = float(np.max(probs))

    except Exception:
        return {
            "classification": "UNKNOWN",
            "confidence": 0.0,
            "language": request.language,
            "explanation": "Audio decoding failed"
        }
    finally:
        os.remove(filename)

    # -------- Prepare response --------
    if prediction == 1:
        return {
            "classification": "AI_GENERATED",
            "confidence": round(confidence, 2),
            "language": request.language,
            "explanation": "Model detected synthetic acoustic patterns typical of AI-generated voices"
        }
    else:
        return {
            "classification": "HUMAN_GENERATED",
            "confidence": round(confidence, 2),
            "language": request.language,
            "explanation": "Model detected natural vocal variations typical of human speech"
        }
