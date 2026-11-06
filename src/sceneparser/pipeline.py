"""
Scene Construction Pipeline
===========================
End-to-end two-stage generative pipeline:
Stage 1: DistilBERT NER Token Classification (Prompt -> Entities JSON)
Stage 2: Conditional Architect Transformer (Entities JSON -> 16x16 Tile Grid)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .model_a.parser import SpriteStackParser
from .model_b.generator import LevelGenerator
from .shared.scene_format import normalize_scene_json, extract_canvas_placements

log = logging.getLogger(__name__)


class SceneConstructionPipeline:
    """
    Two-stage AI scene construction & level generation pipeline.
    """

    def __init__(
        self,
        model_a_path: str | Path | None = None,
        model_b_weights_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.parser = SpriteStackParser(model_path=model_a_path, device=device)
        self.generator = LevelGenerator(weights_path=model_b_weights_path, device=device)

    def parse_only(self, prompt: str) -> dict[str, Any]:
        """Run Stage 1: Parse prompt into normalized entities."""
        raw_json = self.parser.parse_command(prompt)
        normalized = normalize_scene_json(raw_json)
        return normalized

    def generate_only(
        self,
        scene_json: dict[str, Any],
        left_context: list[list[int]] | None = None,
    ) -> list[list[int]]:
        """Run Stage 2: Generate 16x16 tile grid from scene JSON."""
        return self.generator.generate_from_json(scene_json, left_context=left_context)

    def run(
        self,
        prompt: str,
        left_context: list[list[int]] | None = None,
    ) -> dict[str, Any]:
        """
        Run the complete pipeline for a single screen/chunk.

        Parameters
        ----------
        prompt : str
            Natural language scene description.
        left_context : list[list[int]] | None
            Optional 16x8 tile context from previous screen's right edge.

        Returns
        -------
        dict with keys:
          - prompt: str
          - scene_json: dict
          - level_grid: 16x16 list of lists of tile IDs
          - placements: 2D object placements for canvas rendering
          - right_edge: 16x8 right boundary seed for next chunk
          - theme: str
        """
        scene_json = self.parse_only(prompt)
        level_grid = self.generate_only(scene_json, left_context=left_context)
        placements = extract_canvas_placements(scene_json)
        theme = scene_json.get("scene_metadata", {}).get("global_theme", "default")

        return {
            "prompt": prompt,
            "scene_json": scene_json,
            "level_grid": level_grid,
            "placements": placements,
            "theme": theme,
            "right_edge": LevelGenerator.get_right_edge(level_grid),
            "model": "distilbert_ner + architect_transformer",
        }

    def run_wide(self, prompts: list[str]) -> dict[str, Any]:
        """
        Generate a multi-screen continuous wide level from multiple prompts.
        Each screen continues seamlessly from the previous screen's right edge.
        """
        if not prompts:
            return {
                "level_grid": [],
                "scene_jsons": [],
                "placements": [],
                "width": 0,
            }

        # Chunk 0
        first = self.run(prompts[0])
        wide_grid = first["level_grid"]
        scene_jsons = [first["scene_json"]]
        all_placements = list(first["placements"])

        # Subsequent chunks stitched on the right
        num_screens = len(prompts)
        for i, p in enumerate(prompts[1:], start=1):
            context = LevelGenerator.get_right_edge(wide_grid)
            chunk = self.run(p, left_context=context)

            # Shift x coordinates for multi-screen canvas placement
            for item in chunk["placements"]:
                shifted = dict(item)
                shifted["x"] = round((i + item["x"]) / num_screens, 3)
                all_placements.append(shifted)

            wide_grid = LevelGenerator.stitch_grids(wide_grid, chunk["level_grid"])
            scene_jsons.append(chunk["scene_json"])

        return {
            "level_grid": wide_grid,
            "scene_jsons": scene_jsons,
            "placements": all_placements,
            "width": len(wide_grid[0]) if wide_grid else 0,
            "height": 16,
            "right_edge": LevelGenerator.get_right_edge(wide_grid),
        }
