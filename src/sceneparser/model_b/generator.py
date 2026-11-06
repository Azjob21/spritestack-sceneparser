"""
Model B: Level Generator
========================
Autoregressive sampling and level stitching engine for 16x16 tile grids.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from ..shared.scene_format import normalize_scene_json
from .transformer import ConditionalArchitectTransformer

log = logging.getLogger(__name__)

DEFAULT_MODEL_B_PATHS = [
    Path("models/model_b/final_conditional_architect.pth"),
    Path("../api/models/nlp_parser/nlp_parser/model_B/models/final_conditional_architect.pth"),
    Path("api/models/nlp_parser/nlp_parser/model_B/models/final_conditional_architect.pth"),
]


class LevelGenerator:
    """
    Stage 2 Generator: Transforms 27-dim semantic blueprint vectors into
    structured 16x16 tile grids with multi-chunk horizontal stitching.
    """

    def __init__(self, weights_path: str | Path | None = None, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.weights_path = self._resolve_weights_path(weights_path)

        self.model = ConditionalArchitectTransformer(condition_dim=27).to(self.device)
        if self.weights_path and self.weights_path.is_file():
            log.info("Loading Architect Transformer from: %s on %s", self.weights_path, self.device)
            state = torch.load(self.weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            self.is_loaded = True
        else:
            log.warning("Model B weights not found. Running with uninitialized weights.")
            self.is_loaded = False

    @staticmethod
    def _resolve_weights_path(path: str | Path | None) -> Path | None:
        if path:
            p = Path(path)
            if p.is_file():
                return p
        for cand in DEFAULT_MODEL_B_PATHS:
            if cand.is_file():
                return cand.resolve()
        return None

    @staticmethod
    def json_to_blueprint(scene_json: dict[str, Any]) -> torch.Tensor:
        """
        Encode normalized scene JSON into a 27-dimensional condition vector.
        Vector layout:
          Indices 0..8:   Enemy counts in 9 spatial zones
          Indices 9..17:  Loot counts in 9 spatial zones
          Indices 18..26: Player counts in 9 spatial zones
        """
        normalized = normalize_scene_json(scene_json)
        vector = torch.zeros(27, dtype=torch.float32)

        zone_indices = {
            "top-left": 0, "top-center": 1, "top-right": 2,
            "mid-left": 3, "mid-center": 4, "mid-right": 5,
            "bot-left": 6, "bot-center": 7, "bot-right": 8,
        }
        category_offsets = {"enemy": 0, "loot": 9, "player": 18}

        for entity in normalized.get("entities", []):
            zone = entity.get("zone", "mid-center")
            category = entity.get("category", "enemy")
            count = float(int(entity.get("count", 1) or 1))

            if category in category_offsets and zone in zone_indices:
                idx = category_offsets[category] + zone_indices[zone]
                vector[idx] += count

        return vector.unsqueeze(0)

    def generate_level(
        self,
        blueprint: torch.Tensor,
        max_len: int = 256,
        left_context: list[list[int]] | None = None,
    ) -> list[list[int]]:
        """
        Autoregressively sample 256 tile tokens for a 16x16 grid.

        Parameters
        ----------
        blueprint : torch.Tensor
            Shape (1, 27) condition vector.
        max_len : int
            Total sequence length (256 = 16x16).
        left_context : list[list[int]] | None
            Optional 16x8 tile context from right edge of previous chunk.
        """
        generated = torch.tensor([[0]], dtype=torch.long, device=self.device)
        blueprint = blueprint.to(self.device)

        seed_tokens: dict[int, int] = {}
        if left_context is not None:
            for row in range(16):
                for col in range(min(8, len(left_context[row]))):
                    flat_pos = row * 16 + col
                    seed_tokens[flat_pos] = int(left_context[row][col])

        with torch.no_grad():
            for pos in range(max_len - 1):
                actual_pos = pos + 1
                if actual_pos in seed_tokens:
                    next_token = torch.tensor([[seed_tokens[actual_pos]]], dtype=torch.long, device=self.device)
                else:
                    logits = self.model(blueprint, generated)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)

                generated = torch.cat([generated, next_token], dim=1)

        grid = generated.view(16, 16).cpu().tolist()
        return grid

    def generate_from_json(
        self,
        scene_json: dict[str, Any],
        left_context: list[list[int]] | None = None,
    ) -> list[list[int]]:
        """Convenience wrapper: JSON -> 27-dim Blueprint -> 16x16 Grid."""
        blueprint = self.json_to_blueprint(scene_json)
        return self.generate_level(blueprint, left_context=left_context)

    @staticmethod
    def get_right_edge(grid: list[list[int]], cols: int = 8) -> list[list[int]]:
        """Extract rightmost `cols` columns as context seed for the next chunk."""
        return [row[-cols:] for row in grid]

    @staticmethod
    def stitch_grids(
        base_grid: list[list[int]],
        new_grid: list[list[int]],
        overlap: int = 8,
    ) -> list[list[int]]:
        """
        Append a new 16x16 chunk onto the right of an existing grid,
        merging across the `overlap` seed columns.
        """
        return [
            base_row + new_row[overlap:]
            for base_row, new_row in zip(base_grid, new_grid)
        ]
