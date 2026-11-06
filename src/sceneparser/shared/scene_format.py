"""
Shared Scene Normalization and Spatial Formatting
=================================================
Maps natural language positions and entity dictionaries into standardized
zones for Model B and normalized canvas coordinates for SpriteStack Studio.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# 9 spatial zones for the 27-dim Model B condition vector
POSITION_TO_ZONE: dict[str, str] = {
    "left": "mid-left",
    "left-side": "mid-left",
    "center": "mid-center",
    "right": "mid-right",
    "right-side": "mid-right",
    "top-left": "top-left",
    "top-left-corner": "top-left",
    "top-center": "top-center",
    "top": "top-center",
    "top-right": "top-right",
    "top-right-corner": "top-right",
    "mid-left": "mid-left",
    "middle-left": "mid-left",
    "mid-center": "mid-center",
    "middle-center": "mid-center",
    "middle": "mid-center",
    "mid-right": "mid-right",
    "middle-right": "mid-right",
    "bottom-left": "bot-left",
    "bottom-left-corner": "bot-left",
    "bot-left": "bot-left",
    "bottom-center": "bot-center",
    "bot-center": "bot-center",
    "bottom": "bot-center",
    "bottom-floor": "bot-center",
    "floor": "bot-center",
    "ground": "bot-center",
    "bottom-right": "bot-right",
    "bottom-right-corner": "bot-right",
    "bot-right": "bot-right",
}

# 2D normalized (x, y) coordinates for canvas rendering
POSITION_TO_COORDS: dict[str, tuple[float, float]] = {
    "top-left": (0.20, 0.20),
    "top": (0.50, 0.20),
    "top-center": (0.50, 0.20),
    "top-right": (0.80, 0.20),
    "left": (0.20, 0.50),
    "mid-left": (0.20, 0.50),
    "center": (0.50, 0.50),
    "mid-center": (0.50, 0.50),
    "middle": (0.50, 0.50),
    "right": (0.80, 0.50),
    "mid-right": (0.80, 0.50),
    "bottom-left": (0.20, 0.80),
    "bot-left": (0.20, 0.80),
    "bottom": (0.50, 0.80),
    "bot-center": (0.50, 0.80),
    "bottom-right": (0.80, 0.80),
    "bot-right": (0.80, 0.80),
    "far-left": (0.10, 0.50),
    "far-right": (0.90, 0.50),
}

# Object type categorization
OBJECT_TYPES: dict[str, str] = {
    "tree": "stack", "pine": "stack", "oak": "stack", "bush": "stack",
    "rock": "stack", "stone": "stack", "cliff": "stack", "wall": "stack",
    "house": "stack", "castle": "stack", "tower": "stack", "pillar": "stack",
    "grass": "texture", "water": "texture", "sand": "texture", "ground": "texture",
    "knight": "sprite", "hero": "sprite", "player": "sprite", "wizard": "sprite",
    "dragon": "sprite", "skeleton": "sprite", "enemy": "sprite", "slime": "sprite",
    "chest": "sprite", "coin": "sprite", "gem": "sprite", "potion": "sprite",
    "sword": "sprite", "shield": "sprite", "torch": "sprite", "loot": "sprite",
}


def normalize_scene_json(scene_json: dict[str, Any]) -> dict[str, Any]:
    """
    Transform raw Model A NER outputs into standardized Model B input specifications.
    """
    normalized = deepcopy(scene_json)
    entities = normalized.setdefault("entities", [])

    for entity in entities:
        pos_str = str(entity.get("position") or entity.get("zone") or "center").lower().strip()
        obj_name = str(entity.get("object", "enemy")).lower().strip()

        # Zone resolution
        zone = "mid-center"
        if pos_str in POSITION_TO_ZONE:
            zone = POSITION_TO_ZONE[pos_str]
        else:
            best_match = ""
            for key, val in POSITION_TO_ZONE.items():
                if key in pos_str and len(key) > len(best_match):
                    best_match = key
                    zone = val

        # Category resolution (enemy, loot, player)
        category = "other"
        if "player" in obj_name or "hero" in obj_name:
            category = "player"
        elif any(w in obj_name for w in ("loot", "coin", "gem", "treasure", "chest", "item", "lo-ot")):
            category = "loot"
        elif any(w in obj_name for w in ("enemy", "skeleton", "monster", "dragon", "guard", "goblin", "slime", "knight")):
            category = "enemy"

        entity["position"] = pos_str
        entity["zone"] = zone
        entity["category"] = category
        try:
            entity["count"] = max(1, min(int(entity.get("count", 1) or 1), 8))
        except (ValueError, TypeError):
            entity["count"] = 1

    # Extract scene theme
    raw_text = str(normalized.get("scene_metadata", {}).get("raw_text", "")).lower()
    theme = "default"
    if any(w in raw_text for w in ("dungeon", "cave", "castle", "crypt")):
        theme = "dungeon"
    elif any(w in raw_text for w in ("desert", "sand", "pyramid")):
        theme = "desert"
    elif any(w in raw_text for w in ("grassland", "jungle", "forest", "plain", "woods")):
        theme = "grassland"

    normalized.setdefault("scene_metadata", {})["global_theme"] = theme
    return normalized


def extract_canvas_placements(scene_json: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert normalized scene entities into 2D canvas placements with normalized coordinates.
    """
    placements: list[dict[str, Any]] = []
    entities = scene_json.get("entities", [])

    for entity in entities:
        name = str(entity.get("object", "sprite")).replace("-", " ").title()
        obj_type = OBJECT_TYPES.get(name.lower(), "sprite")
        zone = entity.get("zone", "mid-center")
        base_x, base_y = POSITION_TO_COORDS.get(zone, (0.50, 0.50))
        count = int(entity.get("count", 1))

        if count <= 1:
            placements.append({
                "name": name,
                "type": obj_type,
                "x": round(base_x, 3),
                "y": round(base_y, 3),
            })
        else:
            # Spread counted entities horizontally with spacing
            span = min(0.30, 0.08 * (count - 1))
            start_x = max(0.10, base_x - span / 2.0)
            step = span / (count - 1) if count > 1 else 0
            for i in range(count):
                cx = min(0.90, start_x + i * step)
                placements.append({
                    "name": name,
                    "type": obj_type,
                    "x": round(cx, 3),
                    "y": round(base_y, 3),
                })

    return placements
