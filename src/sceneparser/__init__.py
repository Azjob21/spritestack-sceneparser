"""
SpriteStack Scene Parser
========================
Two-stage NLP scene construction & autoregressive level generator pipeline.
"""

from .pipeline import SceneConstructionPipeline
from .model_a.parser import SpriteStackParser
from .model_b.transformer import ConditionalArchitectTransformer
from .model_b.generator import LevelGenerator
from .shared.scene_format import normalize_scene_json, extract_canvas_placements
from .shared.render import level_to_ascii, grid_to_image

__version__ = "0.1.0"

__all__ = [
    "SceneConstructionPipeline",
    "SpriteStackParser",
    "ConditionalArchitectTransformer",
    "LevelGenerator",
    "normalize_scene_json",
    "extract_canvas_placements",
    "level_to_ascii",
    "grid_to_image",
    "__version__",
]
