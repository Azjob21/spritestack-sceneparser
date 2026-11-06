from sceneparser.pipeline import SceneConstructionPipeline


def test_pipeline_run_single():
    pipeline = SceneConstructionPipeline()
    prompt = "Place a player on the left and an enemy in the top-right."
    res = pipeline.run(prompt)

    assert "scene_json" in res
    assert "level_grid" in res
    assert len(res["level_grid"]) == 16
    assert len(res["level_grid"][0]) == 16
    assert "placements" in res
    assert len(res["placements"]) >= 1
    assert "right_edge" in res
    assert len(res["right_edge"]) == 16
    assert len(res["right_edge"][0]) == 8


def test_pipeline_run_wide():
    pipeline = SceneConstructionPipeline()
    prompts = [
        "A grassy plain with a player on the left.",
        "A cave entrance with an enemy guard in the center.",
    ]
    res = pipeline.run_wide(prompts)

    assert res["height"] == 16
    # 16 + (16 - 8) = 24 cols
    assert res["width"] == 24
    assert len(res["scene_jsons"]) == 2
    assert len(res["placements"]) >= 2
