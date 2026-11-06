"""
FastAPI Microservice for SpriteStack Studio Scene Construction
=============================================================
Implements the exact JSON HTTP API contract expected by SpriteStack Studio:
  POST /parse-scene
  GET  /health
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .pipeline import SceneConstructionPipeline

log = logging.getLogger("sceneparser.server")

app = FastAPI(
    title="SpriteStack Scene Parser API",
    description="NLP Scene Construction & Level Generator Service",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: SceneConstructionPipeline | None = None


def get_pipeline() -> SceneConstructionPipeline:
    global _pipeline
    if _pipeline is None:
        log.info("Initializing SceneConstructionPipeline...")
        _pipeline = SceneConstructionPipeline()
    return _pipeline


class ParseSceneRequest(BaseModel):
    prompt: str = Field(default="", description="Natural language prompt")
    left_context: list[list[int]] | None = Field(default=None, description="Optional 16x8 seed grid")
    target_screen: int = Field(default=0, description="Screen index for multi-screen placement")
    num_screens: int = Field(default=1, description="Total number of screens in world")


class ObjectPlacement(BaseModel):
    name: str
    type: str
    x: float
    y: float
    scene_type: str | None = None


class ParseSceneResponse(BaseModel):
    objects: list[ObjectPlacement]
    scene_metadata: dict[str, Any]
    level_grid: list[list[int]] | None = None
    model: str


@app.get("/health")
def health() -> dict[str, Any]:
    pipe = get_pipeline()
    return {
        "status": "ok",
        "service": "spritestack-sceneparser",
        "stage1_transformer_loaded": pipe.parser.is_transformer_loaded,
        "stage2_generator_loaded": pipe.generator.is_loaded,
        "device": str(pipe.generator.device),
    }


@app.post("/parse-scene", response_model=ParseSceneResponse)
def parse_scene(req: ParseSceneRequest) -> dict[str, Any]:
    prompt = req.prompt.strip()
    if not prompt:
        return {
            "objects": [],
            "scene_metadata": {"raw_text": "", "global_theme": "default"},
            "level_grid": [],
            "model": "sceneparser",
        }

    pipe = get_pipeline()
    result = pipe.run(prompt, left_context=req.left_context)

    # Convert placements to ObjectPlacement shape, adjusting for multi-screen if requested
    placements = result["placements"]
    if req.num_screens > 1:
        for p in placements:
            p["x"] = round((req.target_screen + p["x"]) / req.num_screens, 3)

    return {
        "objects": placements,
        "scene_metadata": result["scene_json"].get("scene_metadata", {}),
        "level_grid": result["level_grid"],
        "model": result.get("model", "sceneparser"),
    }
