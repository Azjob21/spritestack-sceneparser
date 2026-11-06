"""
Model A: Natural Language Entity Extractor (NER)
================================================
Fine-tuned DistilBERT Token Classification pipeline for parsing retro game
scene descriptions with fallback support.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import torch
from spellchecker import SpellChecker

log = logging.getLogger(__name__)

# Ensure HuggingFace doesn't attempt network lookups in offline environments
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Candidate search paths for Model A weights
DEFAULT_MODEL_A_PATHS = [
    Path("models/model_a"),
    Path("../api/models/nlp_parser/nlp_parser/model_A/models/SpriteStack_Model_Slim_v2"),
    Path("api/models/nlp_parser/nlp_parser/model_A/models/SpriteStack_Model_Slim_v2"),
]


class SpriteStackParser:
    """
    Stage 1 Parser: Extracts game entities, quantities, positions, and scene types
    from natural language prompts using DistilBERT or fallback heuristics.
    """

    def __init__(self, model_path: str | Path | None = None, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.spell = SpellChecker()
        self.spell.word_frequency.load_words([
            "loot", "climbable", "solid", "platform", "ladder",
            "dungeon", "desert", "grassland", "castle", "jungle", "forest", "cave"
        ])

        self.model = None
        self.tokenizer = None
        self.ner_pipeline = None
        self.is_transformer_loaded = False

        resolved_path = self._resolve_model_path(model_path)
        if resolved_path and (resolved_path / "model.safetensors").is_file():
            self._load_transformer(resolved_path)
        else:
            log.info("Model A weights not found at %s. Using rule-based NER engine.", resolved_path)

    @staticmethod
    def _resolve_model_path(path: str | Path | None) -> Path | None:
        if path:
            p = Path(path)
            if p.exists():
                return p
        for cand in DEFAULT_MODEL_A_PATHS:
            if (cand / "model.safetensors").is_file() or (cand / "config.json").is_file():
                return cand.resolve()
        return None

    def _load_transformer(self, model_path: Path) -> None:
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

            log.info("Loading DistilBERT NER from: %s on %s", model_path, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
            self.tokenizer.model_input_names = [
                n for n in self.tokenizer.model_input_names if n != "token_type_ids"
            ]
            self.model = AutoModelForTokenClassification.from_pretrained(
                str(model_path), local_files_only=True
            ).to(self.device)

            self.ner_pipeline = pipeline(
                "token-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                aggregation_strategy="none",
            )
            self.is_transformer_loaded = True
            log.info("DistilBERT NER loaded successfully.")
        except Exception as e:
            log.warning("Could not load transformer (%s). Falling back to rule-based engine.", e)
            self.is_transformer_loaded = False

    def _sanitize(self, text: str) -> str:
        words = text.lower().split()
        corrected = []
        for w in words:
            # Strip punctuation for spell checking
            clean_w = re.sub(r"[^\w\s]", "", w)
            corr = self.spell.correction(clean_w) if clean_w else w
            corrected.append(corr if corr is not None else w)
        return " ".join(corrected)

    def _glue_with_hyphens(self, raw_results: list[dict[str, Any]]) -> list[dict[str, str]]:
        if not raw_results:
            return []

        merged: list[dict[str, str]] = []
        current_type = raw_results[0]["entity"].split("-")[-1]
        current_word = raw_results[0]["word"].replace("##", "")

        for i in range(1, len(raw_results)):
            next_type = raw_results[i]["entity"].split("-")[-1]
            next_word = raw_results[i]["word"].replace("##", "")

            if next_type == current_type:
                current_word += f"-{next_word}"
            else:
                merged.append({"type": current_type, "text": current_word})
                current_word = next_word
                current_type = next_type

        merged.append({"type": current_type, "text": current_word})
        return merged

    @staticmethod
    def _text_to_int(text: str) -> int:
        num_map = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "a": 1, "an": 1, "single": 1, "couple": 2, "pair": 2,
            "few": 3, "several": 4,
        }
        text_clean = text.lower().strip()
        if text_clean.isdigit():
            return int(text_clean)
        return num_map.get(text_clean, 1)

    def _parse_heuristic(self, text: str) -> dict[str, Any]:
        """Rule-based NER fallback when weights are not present."""
        text_lower = text.lower()
        entities: list[dict[str, Any]] = []

        keywords = {
            "player": ["player", "hero", "knight", "adventurer"],
            "enemy": ["enemy", "skeleton", "goblin", "dragon", "slime", "monster", "bat", "guard"],
            "loot": ["loot", "chest", "coin", "gem", "treasure", "item", "potion"],
            "solid": ["platform", "ground", "wall", "stone", "block", "floor", "brick"],
            "climbable": ["ladder", "vine", "rope"],
        }

        positions = [
            "top-left", "top-right", "top-center", "bottom-left", "bottom-right",
            "bottom-center", "bottom", "top", "left", "right", "center", "middle",
        ]

        # Extract quantities and entities using regex
        for cat, names in keywords.items():
            for name in names:
                pattern = rf"(\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b\s+)?\b{name}s?\b"
                match = re.search(pattern, text_lower)
                if match:
                    cnt_str = match.group(1) or "1"
                    count = self._text_to_int(cnt_str.strip())

                    # Find closest position word
                    pos = "center"
                    for p in positions:
                        if p in text_lower:
                            pos = p
                            break

                    entities.append({
                        "object": name,
                        "count": count,
                        "position": pos,
                        "scene_type": "default",
                    })

        return {
            "entities": entities if entities else [
                {"object": "player", "count": 1, "position": "left", "scene_type": "default"}
            ],
            "scene_metadata": {
                "raw_text": text,
                "engine": "heuristic_fallback",
            },
        }

    def parse_command(self, raw_prompt: str) -> dict[str, Any]:
        """
        End-to-End parse pipeline:
        Spell Check -> Tokenizer -> DistilBERT Inference -> Hyphen Merge -> Schema Construction
        """
        clean_text = self._sanitize(raw_prompt)

        if not self.is_transformer_loaded or self.ner_pipeline is None:
            return self._parse_heuristic(clean_text)

        raw_predictions = self.ner_pipeline(clean_text)
        merged_entities = self._glue_with_hyphens(raw_predictions)

        entities: list[dict[str, Any]] = []
        current_entity: dict[str, Any] = {
            "object": None,
            "count": 1,
            "position": "center",
            "scene_type": "default",
        }
        global_scene_type = "default"

        for item in merged_entities:
            tag_type = item["type"]
            text_val = item["text"]

            if tag_type == "SCENE_TYPE":
                global_scene_type = text_val
            elif tag_type == "COUNT":
                current_entity["count"] = self._text_to_int(text_val)
            elif tag_type == "POSITION":
                current_entity["position"] = text_val
            elif tag_type == "OBJECT":
                current_entity["object"] = text_val
                current_entity["scene_type"] = global_scene_type
                entities.append(current_entity)
                # Reset for next entity
                current_entity = {
                    "object": None,
                    "count": 1,
                    "position": "center",
                    "scene_type": global_scene_type,
                }

        # Apply global scene type if entities lacked it
        for e in entities:
            if e["scene_type"] == "default":
                e["scene_type"] = global_scene_type

        return {
            "entities": entities,
            "scene_metadata": {
                "raw_text": raw_prompt,
                "sanitized_text": clean_text,
                "global_theme": global_scene_type,
                "engine": "distilbert_ner",
            },
        }
