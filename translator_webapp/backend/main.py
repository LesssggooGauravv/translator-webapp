"""
main.py - FastAPI Backend for Seq2Seq Translator

Exposes a single POST /translate endpoint that receives English text
and returns Hindi translation via the trained Seq2Seq model.

Usage:
    cd translator_webapp
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    GET  /health     — Health check
    POST /translate  — Translate English text to Hindi
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path so we can import the ml package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.translate import Translator


# =====================================================================
# REQUEST / RESPONSE SCHEMAS
# =====================================================================

class TranslateRequest(BaseModel):
    """Request body for the /translate endpoint."""
    text: str = Field(..., min_length=1, max_length=500,
                      description="English text to translate")
    src_lang: str = Field(default="en", description="Source language code")
    tgt_lang: str = Field(default="hi", description="Target language code")


class TranslateResponse(BaseModel):
    """Response body from the /translate endpoint."""
    translated_text: str
    src_lang: str
    tgt_lang: str


# =====================================================================
# GLOBAL MODEL INSTANCE (loaded once at startup)
# =====================================================================

translator_instance: Translator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model at startup, clean up on shutdown."""
    global translator_instance
    print("\n--- Loading Translation Model ---")
    try:
        checkpoint_dir = os.path.join(
            os.path.dirname(__file__), "..", "ml", "checkpoints"
        )
        translator_instance = Translator(checkpoint_dir=checkpoint_dir)
        print("--- Model Ready! ---\n")
    except FileNotFoundError as e:
        print(f"\nWARNING: {e}")
        print("The /translate endpoint will return an error until the model is trained.")
        print("Run: python -m ml.train\n")
        translator_instance = None
    yield
    # Cleanup (if needed)
    translator_instance = None


# =====================================================================
# FASTAPI APP
# =====================================================================

app = FastAPI(
    title="Seq2Seq Translator API",
    description="English to Hindi translation using Attention-based Seq2Seq model",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the local frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],             # Allow all origins for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# ENDPOINTS
# =====================================================================

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": translator_instance is not None,
    }


@app.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    """
    Translate English text to Hindi.
    
    Receives the input text, passes it through the Seq2Seq model,
    and returns the translated Hindi string.
    """
    # Check if model is loaded
    if translator_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first by running: "
                   "python -m ml.train"
        )

    # Validate language pair (currently only en -> hi is supported)
    if request.src_lang != "en" or request.tgt_lang != "hi":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language pair: {request.src_lang} -> {request.tgt_lang}. "
                   "Currently only 'en' -> 'hi' is supported."
        )

    try:
        # Run inference
        translated = translator_instance.translate(request.text)

        return TranslateResponse(
            translated_text=translated,
            src_lang=request.src_lang,
            tgt_lang=request.tgt_lang,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation error: {str(e)}"
        )
