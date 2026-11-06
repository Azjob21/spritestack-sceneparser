from sceneparser.server import health, parse_scene, ParseSceneRequest


def test_health_endpoint():
    res = health()
    assert isinstance(res, dict)
    assert res["status"] == "ok"
    assert res["service"] == "spritestack-sceneparser"


def test_parse_scene_endpoint():
    req = ParseSceneRequest(
        prompt="A dungeon with 2 knights on the left and a treasure chest on the right.",
        num_screens=1,
    )
    res = parse_scene(req)

    assert "objects" in res
    assert len(res["objects"]) >= 2
    assert "level_grid" in res
    assert len(res["level_grid"]) == 16
    assert "scene_metadata" in res
