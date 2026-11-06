"""
Basic example for Scene Construction Pipeline.
"""

from sceneparser.pipeline import SceneConstructionPipeline
from sceneparser.shared.render import level_to_ascii, grid_to_image


def main():
    pipeline = SceneConstructionPipeline()
    prompt = "A dark dungeon with a player on the left, two skeletons in the center, and a treasure chest on the right."

    print(f"Prompt: {prompt}\n")
    result = pipeline.run(prompt)

    print("Parsed Entities:")
    for e in result["scene_json"]["entities"]:
        print(f"  • {e['count']}x {e['object']} in {e['zone']} ({e['category']})")

    print("\n16x16 Generated Tile Grid:")
    print(level_to_ascii(result["level_grid"]))

    # Save PNG rendering
    img = grid_to_image(result["level_grid"], theme=result["theme"])
    img.save("basic_dungeon.png")
    print("\nSaved visual render to basic_dungeon.png")


if __name__ == "__main__":
    main()
