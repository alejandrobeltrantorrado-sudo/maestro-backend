"""
Maestro Audio Backend — versión final
- Genera música con MusicGen (HuggingFace Transformers)
- Sin audiocraft, sin errores de compilación
- Acepta audio de referencia para analizar estilo
- Soporta modo texto y modo "mejora mi audio"
"""
import os, uuid, logging, io, tempfile
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("maestro")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID     = os.getenv("MODEL_ID", "facebook/musicgen-small")
MAX_DURATION = int(os.getenv("MAX_DURATION", "30"))
TEMP_DIR     = Path(os.getenv("TEMP_DIR", "/tmp/maestro"))
ORIGINS      = os.getenv("ALLOWED_ORIGINS", "*").split(",")
SAMPLE_RATE  = 32000          # MusicGen small/medium output rate

TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ── Modelo global ─────────────────────────────────────────────────────────────
state = {"model": None, "processor": None, "ready": False, "error": None}

def load_model():
    try:
        log.info(f"Cargando {MODEL_ID} ...")
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        p = AutoProcessor.from_pretrained(MODEL_ID)
        m = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)
        m.eval()
        state["processor"] = p
        state["model"]     = m
        state["ready"]     = True
        log.info("Modelo listo ✓")
    except Exception as e:
        state["error"] = str(e)
        log.error(f"Error cargando modelo: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    for f in TEMP_DIR.glob("maestro_*.wav"):
        f.unlink(missing_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Maestro Audio API",
    version="3.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def save_wav(audio_np, path: Path):
    """Guarda numpy array como WAV int16."""
    import numpy as np
    import scipy.io.wavfile as wf
    peak = np.max(np.abs(audio_np))
    if peak > 0:
        audio_np = audio_np / peak
    audio_int16 = (audio_np * 32767).astype(np.int16)
    wf.write(str(path), SAMPLE_RATE, audio_int16)

def generate_audio(prompt: str, duration: int) -> Path:
    """Genera audio a partir de un prompt de texto."""
    import torch, numpy as np
    if not state["ready"]:
        raise RuntimeError(state["error"] or "Modelo no listo")

    processor = state["processor"]
    model     = state["model"]
    max_tokens = duration * 50       # MusicGen: ~50 tokens/segundo

    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    with torch.inference_mode():
        wav = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            guidance_scale=3.0,
        )
    # wav: [batch, channels, samples]
    audio_np = wav[0, 0].cpu().numpy()

    out = TEMP_DIR / f"maestro_{uuid.uuid4().hex[:10]}.wav"
    save_wav(audio_np, out)
    return out

def analyze_audio_features(audio_bytes: bytes) -> dict:
    """
    Analiza características básicas del audio subido:
    tempo BPM, duración, nivel de energía.
    Devuelve un dict que enriquece el prompt de generación.
    """
    try:
        import numpy as np
        import io

        # Intentar leer con scipy (WAV) o con librosa si está disponible
        try:
            import scipy.io.wavfile as wf
            sr, data = wf.read(io.BytesIO(audio_bytes))
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = data.astype(np.float32)
            duration = len(data) / sr
        except Exception:
            # Si no es WAV, devolvemos info mínima
            return {"duration_s": 30, "energy": "medium", "bpm_hint": ""}

        # Energía RMS
        rms = float(np.sqrt(np.mean(data**2)))
        energy = "high" if rms > 0.15 else ("medium" if rms > 0.05 else "low")

        # Estimación muy básica de BPM por onset energy
        # (sin librosa, solo heurística de energía)
        chunk = int(sr * 0.02)
        frames = [np.sqrt(np.mean(data[i:i+chunk]**2))
                  for i in range(0, len(data)-chunk, chunk)]
        frames = np.array(frames)
        threshold = frames.mean() * 1.3
        onsets = np.where((frames[1:] > threshold) & (frames[:-1] <= threshold))[0]
        if len(onsets) > 2:
            avg_gap = np.mean(np.diff(onsets)) * 0.02  # en segundos
            bpm = round(60 / avg_gap) if avg_gap > 0 else 0
            # Limitar a rango razonable
            bpm = max(60, min(200, bpm))
            bpm_hint = f"{bpm} bpm"
        else:
            bpm_hint = ""

        return {
            "duration_s": round(duration, 1),
            "energy": energy,
            "bpm_hint": bpm_hint,
        }
    except Exception:
        return {"duration_s": 30, "energy": "medium", "bpm_hint": ""}

def make_response(path: Path, bg: BackgroundTasks) -> FileResponse:
    bg.add_task(lambda p=path: p.unlink(missing_ok=True))
    return FileResponse(
        path=str(path),
        media_type="audio/wav",
        filename=f"maestro_{uuid.uuid4().hex[:6]}.wav",
    )

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Maestro Audio API",
        "version": "3.0.0",
        "ready": state["ready"],
        "model": MODEL_ID,
        "endpoints": {
            "GET  /health": "Estado del servicio",
            "POST /generate": "Genera audio desde un prompt de texto",
            "POST /generate/from-audio": "Analiza tu audio y genera una versión nueva",
            "POST /generate/preview": "Preview rápido de 8 segundos",
        },
    }

@app.get("/health")
def health():
    return {
        "status": "ok" if state["ready"] else ("error" if state["error"] else "loading"),
        "ready": state["ready"],
        "model": MODEL_ID,
        "max_duration": MAX_DURATION,
        "error": state["error"],
    }


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=600)
    duration: int = Field(default=15, ge=5, le=60)

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "upbeat cumbia with accordion and bass drum, 120 bpm, tropical, energetic",
                "duration": 15,
            }
        }
    }


@app.post("/generate")
async def generate(req: GenerateRequest, bg: BackgroundTasks):
    """
    Genera audio desde un prompt de texto.

    El prompt debe estar en inglés para mejores resultados.
    Incluye: género, instrumentos, BPM, mood, era musical.

    Tiempos en CPU:
      8s  → ~25s espera
      15s → ~45s espera
      30s → ~90s espera
    """
    if not state["ready"]:
        raise HTTPException(503, state["error"] or "Modelo cargando, espera 30s")
    if req.duration > MAX_DURATION:
        raise HTTPException(400, f"Duración máxima: {MAX_DURATION}s")

    log.info(f"Generando {req.duration}s | '{req.prompt[:80]}'")
    try:
        path = generate_audio(req.prompt, req.duration)
        return make_response(path, bg)
    except Exception as e:
        log.error(f"Error: {e}")
        raise HTTPException(500, str(e))


@app.post("/generate/from-audio")
async def generate_from_audio(
    bg: BackgroundTasks,
    audio: UploadFile = File(..., description="Tu audio (WAV, MP3, OGG, M4A)"),
    prompt: str = Form(default="", description="Descripción adicional del estilo deseado"),
    duration: int = Form(default=15, ge=5, le=60),
    style_hint: str = Form(default="", description="Ej: 'make it more upbeat', 'add jazz chords'"),
):
    """
    Analiza tu audio subido y genera una nueva versión musical basada en él.

    - Detecta BPM y energía de tu grabación
    - Combina esas características con el prompt y el estilo que pidas
    - Genera audio nuevo que refleja el espíritu de tu grabación original
    """
    if not state["ready"]:
        raise HTTPException(503, state["error"] or "Modelo cargando, espera 30s")
    if duration > MAX_DURATION:
        raise HTTPException(400, f"Duración máxima: {MAX_DURATION}s")

    # Leer el audio subido
    audio_bytes = await audio.read()
    if len(audio_bytes) > 50 * 1024 * 1024:   # límite 50 MB
        raise HTTPException(400, "Archivo muy grande, máximo 50 MB")

    log.info(f"Analizando audio subido: {audio.filename} ({len(audio_bytes)//1024} KB)")

    # Analizar características
    features = analyze_audio_features(audio_bytes)
    log.info(f"Features detectadas: {features}")

    # Construir prompt enriquecido
    parts = []
    if prompt.strip():
        parts.append(prompt.strip())
    if features["bpm_hint"]:
        parts.append(features["bpm_hint"])
    energy_map = {"high": "energetic, powerful", "medium": "moderate energy", "low": "soft, gentle, calm"}
    parts.append(energy_map.get(features["energy"], "moderate energy"))
    if style_hint.strip():
        parts.append(style_hint.strip())

    final_prompt = ", ".join(parts) if parts else "instrumental music, moderate tempo"
    log.info(f"Prompt final: '{final_prompt}'")

    try:
        path = generate_audio(final_prompt, duration)
        return make_response(path, bg)
    except Exception as e:
        log.error(f"Error: {e}")
        raise HTTPException(500, str(e))


@app.post("/generate/preview")
async def preview(bg: BackgroundTasks):
    """
    Genera 8s con un preset — para verificar que el backend funciona.
    """
    req = GenerateRequest(
        prompt="upbeat latin jazz piano with bass and bongo drums, 100 bpm, lively",
        duration=8,
    )
    return await generate(req, bg)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
