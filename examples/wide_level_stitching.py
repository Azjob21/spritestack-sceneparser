"""
Multi-Screen Infinite Level Stitching Example.
"""

from sceneparser.pipeline import SceneConstructionPipeline
from sceneparser.shared.render import level_to_ascii, grid_to_image


def main():
    pipeline = SceneConstructionPipeline()
    prompts = [
        "A grassland valley with a player on the left.",
        "A rocky transition with two goblins on a platform.",
        "A castle gate with a treasure chest and a dragon guard.",
    ]

    print(f"Generating 3-chunk continuous level across {len(prompts)} screens...\n")
    result = pipeline.run_wide(prompts)

    print(f"Wide Level Dimension: {result['width']} cols x {result['height']} rows")
    print(level_to_ascii(result["level_grid"]))

    img = grid_to_image(result["level_grid"], theme="grassland")
    img.save("wide_level_3screens.png")
    print("\nSaved stitched multi-screen level render to wide_level_3screens.png")


if __name__ == "__main__":
    main()
