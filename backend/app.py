from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from backend.models import MODEL_NAMES, load_models_and_reports, predict

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models, app.state.report = load_models_and_reports()
    yield


app = FastAPI(
    title="Brain MRI Research Dashboard API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    loaded_models = getattr(app.state, "models", {})
    report = getattr(app.state, "report", {})
    return {
        "status": "ready" if len(loaded_models) == len(MODEL_NAMES) else "not_ready",
        "models": list(loaded_models),
        "run_id": report.get("run_id"),
        "device": report.get("device"),
    }


@app.get("/api/models")
async def get_models() -> dict[str, Any]:
    report = getattr(app.state, "report", None)
    if report is None:
        raise HTTPException(status_code=503, detail="Model reports are not ready.")
    return report


@app.post("/api/predict")
async def predict_image(
    model_name: str = Form(...),
    image: UploadFile = File(...),
) -> dict[str, Any]:
    loaded_models = getattr(app.state, "models", {})
    report = getattr(app.state, "report", {})
    if model_name not in MODEL_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"Choose one of: {', '.join(MODEL_NAMES)}.",
        )
    if model_name not in loaded_models:
        raise HTTPException(status_code=503, detail=f"{model_name} is not loaded.")

    content = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")
    try:
        with Image.open(BytesIO(content)) as opened_image:
            if opened_image.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(
                    status_code=415,
                    detail="Upload a JPEG, PNG, or WebP image.",
                )
            if opened_image.width * opened_image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions are too large to process.",
                )
            opened_image.load()
            image_rgb = opened_image.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(
            status_code=422,
            detail="The uploaded file is not a valid supported image.",
        ) from error

    try:
        result = predict(
            image_rgb,
            model_name,
            loaded_models[model_name],
            report["class_names"],
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error

    result["run_id"] = report["run_id"]
    result["filename"] = image.filename
    result["benchmark"] = next(
        item["test_report"]
        for item in report["models"]
        if item["name"] == model_name
    )
    return result
