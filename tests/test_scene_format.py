from sceneparser.shared.scene_format import (
    normalize_scene_json,
    extract_canvas_placements,
    POSITION_TO_ZONE,
)


def test_position_to_zone_mapping():
    assert POSITION_TO_ZONE["left"] == "mid-left"
    assert POSITION_TO_ZONE["top-right"] == "top-right"
    assert POSITION_TO_ZONE["bottom-left"] == "bot-left"
    assert POSITION_TO_ZONE["center"] == "mid-center"


def test_normalize_scene_json():
    raw = {
        "entities": [
            {"object": "player", "count": "1", "position": "bottom-left"},
            {"object": "dragon", "count": 2, "position": "top-right"},
            {"object": "treasure", "count": "1", "position": "middle"},
        ],
        "scene_metadata": {"raw_text": "a dark dungeon with a player"},
    }

    normalized = normalize_scene_json(raw)
    entities = normalized["entities"]
    assert len(entities) == 3

    # Player checks
    assert entities[0]["category"] == "player"
    assert entities[0]["zone"] == "bot-left"
    assert entities[0]["count"] == 1

    # Enemy checks
    assert entities[1]["category"] == "enemy"
    assert entities[1]["zone"] == "top-right"
    assert entities[1]["count"] == 2

    # Loot checks
    assert entities[2]["category"] == "loot"
    assert entities[2]["zone"] == "mid-center"

    # Theme check
    assert normalized["scene_metadata"]["global_theme"] == "dungeon"


def test_extract_canvas_placements():
    scene = {
        "entities": [
            {"object": "knight", "count": 2, "zone": "mid-left"},
            {"object": "chest", "count": 1, "zone": "mid-right"},
        ]
    }
    placements = extract_canvas_placements(scene)
    assert len(placements) == 3  # 2 knights + 1 chest
    assert placements[0]["name"] == "Knight"
    assert placements[2]["name"] == "Chest"
    for p in placements:
        assert 0.0 <= p["x"] <= 1.0
        assert 0.0 <= p["y"] <= 1.0
