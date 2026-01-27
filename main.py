from fastapi import FastAPI, Header, HTTPException, Request
from datetime import datetime

app = FastAPI(title="Agentic Honeypot API")

API_KEY = "123456"

@app.get("/")
def home():
    return {
        "status": "Honeypot service active",
        "message": "Unauthorized access attempts are monitored"
    }

# ------------------ HONEYPOT ENDPOINT ------------------
@app.post("/honeypot")
async def honeypot_endpoint(
    request: Request,
    x_api_key: str = Header(None)
):
    # --- AUTH CHECK ---
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # --- CAPTURE REQUEST METADATA ---
    client_ip = request.client.host if request.client else "unknown"
    headers = dict(request.headers)
    timestamp = datetime.utcnow().isoformat()

    # (In real honeypot, this would be stored / analyzed)
    honeypot_event = {
        "timestamp": timestamp,
        "client_ip": client_ip,
        "headers_captured": True,
        "activity_flagged": True
    }

    # --- DECOY RESPONSE ---
    return {
        "status": "access_logged",
        "severity": "low",
        "message": "Request received and monitored",
        "honeypot_event": honeypot_event
    }
