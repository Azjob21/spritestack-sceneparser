"""
Tilemap Visualizer & Renderer
=============================
Renders 16xN tile grids into ASCII art or high-resolution pixel-art PNG images.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

TILE_NAMES = {
    0: "Air",
    1: "Solid",
    2: "Loot",
    3: "Enemy",
    4: "Climbable",
    5: "Player",
}

# Safe ASCII character mappings for cross-platform terminals
ASCII_CHAR_MAP = {
    0: "  ",   # Air
    1: "##",   # Solid block
    2: "$$",   # Loot / Chest
    3: "XX",   # Enemy
    4: "||",   # Climbable ladder
    5: "@@",   # Player
}

# Color palette for 2D PNG tilemap rendering (RGBA)
TILE_COLORS = {
    0: (20, 24, 34, 255),       # Sky / Dark Dungeon Air
    1: (110, 115, 130, 255),    # Solid Stone Blocks
    2: (255, 215, 0, 255),      # Golden Loot Chest
    3: (220, 40, 50, 255),      # Red Enemy Mark
    4: (180, 120, 50, 255),     # Wooden Ladder / Climbable
    5: (50, 205, 50, 255),      # Green Player Hero
}


def level_to_ascii(grid: list[list[int]]) -> str:
    """Render any-width tile grid as clean ASCII text."""
    if not grid:
        return ""
    width = len(grid[0]) * 2 + 2
    lines = ["+" + "-" * (width - 2) + "+"]
    for row in grid:
        line = "".join(ASCII_CHAR_MAP.get(int(cell), "??") for cell in row)
        lines.append(f"|{line}|")
    lines.append("+" + "-" * (width - 2) + "+")
    return "\n".join(lines)


def grid_to_image(
    grid: list[list[int]],
    tile_size: int = 16,
    theme: str = "dungeon",
) -> Image.Image:
    """
    Render 16xN tile grid into an RGBA PIL Image.

    Parameters
    ----------
    grid : list of list of int
        2D grid where grid[row][col] is the tile ID.
    tile_size : int
        Size in pixels of each tile.
    theme : str
        Color theme ('dungeon', 'grassland', 'desert').
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    img = Image.new("RGBA", (cols * tile_size, rows * tile_size), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)

    theme_bg = {
        "dungeon": (15, 18, 25, 255),
        "grassland": (70, 140, 210, 255),
        "desert": (220, 190, 130, 255),
    }.get(theme.lower(), (15, 18, 25, 255))

    theme_solid = {
        "dungeon": (85, 90, 105, 255),
        "grassland": (75, 145, 60, 255),
        "desert": (180, 130, 70, 255),
    }.get(theme.lower(), (85, 90, 105, 255))

    for r in range(rows):
        for c in range(cols):
            tile = int(grid[r][c])
            x0 = c * tile_size
            y0 = r * tile_size
            x1 = x0 + tile_size
            y1 = y0 + tile_size

            # Base background
            if tile == 0:
                draw.rectangle([x0, y0, x1, y1], fill=theme_bg)
            elif tile == 1:
                # Solid block with border highlight
                draw.rectangle([x0, y0, x1, y1], fill=theme_solid)
                draw.rectangle([x0, y0, x1 - 1, y0 + 1], fill=(255, 255, 255, 60))
                draw.rectangle([x0, y1 - 2, x1, y1 - 1], fill=(0, 0, 0, 80))
            elif tile == 2:
                # Loot chest
                draw.rectangle([x0, y0, x1, y1], fill=theme_bg)
                draw.rectangle([x0 + 2, y0 + 4, x1 - 2, y1 - 2], fill=(255, 200, 30, 255), outline=(180, 130, 10, 255))
            elif tile == 3:
                # Enemy
                draw.rectangle([x0, y0, x1, y1], fill=theme_bg)
                draw.rectangle([x0 + 3, y0 + 3, x1 - 3, y1 - 3], fill=(230, 45, 55, 255), outline=(130, 10, 20, 255))
            elif tile == 4:
                # Ladder
                draw.rectangle([x0, y0, x1, y1], fill=theme_bg)
                draw.line([(x0 + 3, y0), (x0 + 3, y1)], fill=(160, 110, 50, 255), width=2)
                draw.line([(x1 - 4, y0), (x1 - 4, y1)], fill=(160, 110, 50, 255), width=2)
                for step_y in range(y0 + 3, y1, 4):
                    draw.line([(x0 + 3, step_y), (x1 - 4, step_y)], fill=(160, 110, 50, 255), width=1)
            elif tile == 5:
                # Player
                draw.rectangle([x0, y0, x1, y1], fill=theme_bg)
                draw.rectangle([x0 + 3, y0 + 2, x1 - 3, y1 - 2], fill=(60, 220, 80, 255), outline=(20, 120, 30, 255))

    return img
