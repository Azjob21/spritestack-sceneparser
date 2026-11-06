import torch
from sceneparser.model_b.transformer import ConditionalArchitectTransformer
from sceneparser.model_b.generator import LevelGenerator


def test_transformer_forward():
    model = ConditionalArchitectTransformer(condition_dim=27, vocab_size=6, d_model=64, num_layers=2)
    model.eval()

    cond = torch.zeros(1, 27)
    src = torch.zeros(1, 16, dtype=torch.long)

    with torch.no_grad():
        out = model(cond, src)
    assert out.shape == (1, 16, 6)


def test_generator_blueprint_encoding():
    scene = {
        "entities": [
            {"object": "player", "category": "player", "zone": "bot-left", "count": 1},
            {"object": "enemy", "category": "enemy", "zone": "top-right", "count": 3},
        ]
    }
    vec = LevelGenerator.json_to_blueprint(scene)
    assert vec.shape == (1, 27)

    # Enemy in top-right: offset 0 + index 2 = 2
    assert vec[0, 2].item() == 3.0
    # Player in bot-left: offset 18 + index 6 = 24
    assert vec[0, 24].item() == 1.0


def test_generator_stitching():
    # 16x16 grid A
    grid_a = [[0] * 16 for _ in range(16)]
    # 16x16 grid B
    grid_b = [[1] * 16 for _ in range(16)]

    stitched = LevelGenerator.stitch_grids(grid_a, grid_b, overlap=8)
    assert len(stitched) == 16
    # 16 + (16 - 8) = 24 columns
    assert len(stitched[0]) == 24
