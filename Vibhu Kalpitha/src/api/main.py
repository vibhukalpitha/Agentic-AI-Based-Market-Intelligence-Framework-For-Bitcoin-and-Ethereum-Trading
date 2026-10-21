import os
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add Vibhu Kalpitha root to sys.path so modules import reliably
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DIST_DIR = os.path.join(PROJECT_ROOT, "frontend", "dist")
SIMILARITY_IMG_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "component_01", "results", "similarity"))

from datetime import datetime, timezone
from src.api.inference_service import run_full_prediction, registry
from src.api.regime_service import get_regime_snapshot

app = FastAPI(
    title="Agentic AI Market Intelligence API (BTC & ETH)",
    description="Multi-Horizon (1H & 4H) Market Intelligence & Prediction Framework",
    version="1.0.0"
)

# Mount Component 01 DTW similarity visual charts
if os.path.exists(SIMILARITY_IMG_DIR):
    app.mount("/api/similarity-images", StaticFiles(directory=SIMILARITY_IMG_DIR), name="similarity_images")

# Enable CORS for local Vite development & custom ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    trade_date: Optional[str] = None
    trade_time: Optional[str] = None
    is_live: bool = True

@app.on_event("startup")
async def startup_event():
    print("[API] Initializing API: Pre-loading all 1H & 4H models...")
    try:
        registry.load_all()
        print("[API] Models ready for inference requests.")
    except Exception as e:
        print(f"[WARN] Startup warning during model loading: {e}")

if os.path.exists(DIST_DIR):
    assets_dir = os.path.join(DIST_DIR, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    def serve_react_app():
        index_file = os.path.join(DIST_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {
            "framework": "Agentic AI Market Intelligence Framework",
            "author": "Vibhu Kalpitha",
            "supported_coins": ["BTC", "ETH"],
            "horizons": ["1-Hour", "4-Hour"],
            "status": "online"
        }
else:
    @app.get("/")
    def root():
        return {
            "framework": "Agentic AI Market Intelligence Framework",
            "author": "Vibhu Kalpitha",
            "supported_coins": ["BTC", "ETH"],
            "horizons": ["1-Hour", "4-Hour"],
            "status": "online"
        }

@app.get("/api/health")
def health_check():
    return {"status": "ok", "models_loaded": registry.loaded}

@app.get("/api/live")
def get_live_prediction():
    try:
        result = run_full_prediction(is_live=True)
        now_utc = datetime.now(timezone.utc)
        result["regimes"] = {
            "BTC": get_regime_snapshot("BTCUSDT", now_utc, is_live=True),
            "ETH": get_regime_snapshot("ETHUSDT", now_utc, is_live=True)
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
def predict_market(req: PredictionRequest):
    try:
        result = run_full_prediction(
            trade_date=req.trade_date,
            trade_time=req.trade_time,
            is_live=req.is_live
        )
        target_dt = datetime.now(timezone.utc)
        if not req.is_live and req.trade_date and req.trade_time:
            try:
                target_dt = datetime.strptime(f"{req.trade_date} {req.trade_time}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            except Exception:
                pass

        result["regimes"] = {
            "BTC": get_regime_snapshot("BTCUSDT", target_dt, is_live=req.is_live),
            "ETH": get_regime_snapshot("ETHUSDT", target_dt, is_live=req.is_live)
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
