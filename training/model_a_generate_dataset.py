import json
import itertools
import random

# 1. Define the Ontology (Vocabulary)
SCENES = [
    "dungeon", "forest", "castle", "cave", "village", 
    "desert", "graveyard", "swamp", "tavern", "island"
]

OBJECTS = [
    "player",
    "skeleton",
    "goblin",
    "dragon",
    "villager",
    "merchant",
    "knight",
    "wizard",
    "slime",
    "ghost",
    "bat",
    "chest",
    "coin",
    "potion",
    "sword",
    "shield",
    "bow",
    "arrow",
    "key",
    "map",
    "gem",
    "scroll",
    "ring",
    "crown",
    "tree",
    "rock",
    "bush",
    "flower",
    "stump",
    "river",
    "pond",
    "log",
    "vine",
    "mushroom",
    "door",
    "window",
    "bed",
    "table",
    "chair",
    "bookshelf",
    "barrel",
    "crate",
    "anvil",
    "forge",
    "torch",
    "campfire",
    "lantern",
    "candle",
    "statue",
    "painting",
    "rug",
    "fountain",
]

POSITIONS = [
    "left", "right", "center", "top", "bottom", 
    "top-left", "top-right", "bottom-left", "bottom-right", 
    "foreground", "background", "middle"
]

# --- NEW: Number Vocabulary (1 to 20) ---
NUMBERS = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
]

# 2. SENTENCE TEMPLATES (Updated with {count} support)
TEMPLATES = [
    # --- Category 1: Direct Commands with Counts ---
    [
        ("spawn", "O"),
        ("{count}", "B-COUNT"),
        ("{object1}", "B-OBJECT"),
        ("at the", "O"),
        ("{position1}", "B-POSITION"),
    ],
    [
        ("place", "O"),
        ("{count}", "B-COUNT"),
        ("{object1}", "B-OBJECT"),
        ("in the", "O"),
        ("{scene}", "B-SCENE_TYPE"),
    ],
    [
        ("add", "O"),
        ("{count}", "B-COUNT"),
        ("{object1}", "B-OBJECT"),
        ("{position1}", "B-POSITION"),
    ],
    # --- Category 2: Multiple Objects in One Scene ---
    [
        ("in the", "O"),
        ("{scene}", "B-SCENE_TYPE"),
        ("put", "O"),
        ("{count}", "B-COUNT"),
        ("{object1}", "B-OBJECT"),
        ("on the", "O"),
        ("{position1}", "B-POSITION"),
    ],
    # --- Category 3: Conversational ---
    [
        ("i want", "O"),
        ("{count}", "B-COUNT"),
        ("{object1}", "B-OBJECT"),
        ("somewhere near the", "O"),
        ("{position1}", "B-POSITION"),
    ],
    [
        ("can we have", "O"),
        ("{count}", "B-COUNT"),
        ("{object1}", "B-OBJECT"),
        ("standing in the", "O"),
        ("{scene}", "B-SCENE_TYPE"),
    ],
    # --- Category 4: Minimalist ---
    [("{count}", "B-COUNT"), ("{object1}", "B-OBJECT"), ("{position1}", "B-POSITION")],
]


def generate_dataset(output_file, target_size=2000):
    dataset = []

    for i in range(target_size):
        # Pick random components
        scene = random.choice(SCENES)
        obj1 = random.choice(OBJECTS)
        pos1 = random.choice(POSITIONS)
        count = random.choice(NUMBERS)

        template = random.choice(TEMPLATES)

        tokens = []
        ner_tags = []

        for phrase, tag in template:
            # Replace placeholders
            text = phrase.format(scene=scene, object1=obj1, position1=pos1, count=count)
            words = text.split()

            for j, word in enumerate(words):
                tokens.append(word)
                if tag == "O":
                    ner_tags.append("O")
                elif j == 0:
                    ner_tags.append(tag)
                else:
                    # Logic for I- tags (e.g., I-OBJECT or I-POSITION)
                    ner_tags.append(tag.replace("B-", "I-"))

        dataset.append({
            "tokens": tokens,
            "ner_tags": ner_tags
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')

    print(f"✅ Generated {len(dataset)} sentences with B-COUNT tags in {output_file}")


if __name__ == "__main__":
    # Increased target_size to 5000 for better learning of numbers
    generate_dataset("data/train.jsonl", target_size=5000)
