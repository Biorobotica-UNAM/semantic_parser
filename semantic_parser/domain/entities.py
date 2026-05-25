"""
domain/entities.py

Single source of truth for every entity the robot knows about.

Design notes
────────────
• Open/Closed Principle: add a new category by extending the dicts/sets below
  — no other module changes.
• Each constant is a frozenset or tuple so it is immutable and hashable.
• Naming follows the corpus of GPSR commands used by the robot.

BioRobotics Laboratory, Facultad de Ingeniería, UNAM
"""

from typing import Dict, FrozenSet, List, Tuple

# ──────────────────────────────────────────────────────────────
# LOCATIONS
# ──────────────────────────────────────────────────────────────

# Multi-word locations must be listed longest-first so the
# EntityRuler's phrase-matcher matches the full span before a
# sub-span (e.g. "kitchen table" before "kitchen").
LOCATIONS: Tuple[str, ...] = (
    "bedside table",
    "kitchen table",
    "side tables",
    "potted plant",
    "storage rack",
    "desk lamp",
    "living room",
    "waste basket",
    "coatrack",
    "trashbin",
    "bookshelf",
    "dishwasher",
    "refrigerator",
    "armchair",
    "bathroom",
    "bedroom",
    "entrance",
    "kitchen",
    "pantry",
    "office",
    "chairs",
    "cabinet",
    "shelf",
    "sofa",
    "sink",
    "desk",
    "lamp",
    "exit",
    "bed",
)

# ──────────────────────────────────────────────────────────────
# OBJECT CATEGORIES
# ──────────────────────────────────────────────────────────────

OBJECT_CATEGORIES: Tuple[str, ...] = (
    "cleaning supply",
    "orange juice",
    "tropical juice",
    "strawberry jello",
    "chocolate jello",
    "coffee grounds",
    "red wine",
    "iced tea",
    "juice pack",
    "cornflakes",
    "pringles",
    "cheezit",
    "cleanser",
    "mustard",
    "sponge",
    "banana",
    "orange",
    "tomato soup",
    "soccer ball",
    "tennis ball",
    "baseball",
    "rubiks cube",
    "sugar",
    "spoon",
    "plate",
    "knife",
    "fork",
    "bowl",
    "cup",
    "tuna",
    "spam",
    "cola",
    "milk",
    "pear",
    "plum",
    "peach",
    "lemon",
    "apple",
    "dice",
    "drink",
    "snack",
    "fruit",
    "food",
    "dish",
    "toy",
)

# Semantic grouping: maps an object string to its abstract category.
# Used downstream by the CLIPS planner to reason over types.
OBJECT_CATEGORY_MAP: Dict[str, str] = {
    "banana": "fruit",
    "orange": "fruit",
    "pear": "fruit",
    "plum": "fruit",
    "peach": "fruit",
    "lemon": "fruit",
    "apple": "fruit",
    "tomato soup": "food",
    "spam": "food",
    "tuna": "food",
    "cornflakes": "food",
    "mustard": "food",
    "sugar": "food",
    "cola": "drink",
    "milk": "drink",
    "juice pack": "drink",
    "orange juice": "drink",
    "tropical juice": "drink",
    "iced tea": "drink",
    "red wine": "drink",
    "coffee grounds": "drink",
    "pringles": "snack",
    "cheezit": "snack",
    "strawberry jello": "snack",
    "chocolate jello": "snack",
    "soccer ball": "toy",
    "tennis ball": "toy",
    "baseball": "toy",
    "rubiks cube": "toy",
    "dice": "toy",
    "spoon": "dish",
    "plate": "dish",
    "fork": "dish",
    "bowl": "dish",
    "cup": "dish",
    "knife": "dish",
    "sponge": "cleaning supply",
    "cleanser": "cleaning supply",
}

# ──────────────────────────────────────────────────────────────
# PERSON NAMES
# ──────────────────────────────────────────────────────────────

PERSON_NAMES: FrozenSet[str] = frozenset(
    {
        "Angel",
        "Robin",
        "Paris",
        "Morgan",
        "Charlie",
        "Jules",
        "Adel",
        "Axel",
        "Jane",
        "Simone",
    }
)

# ──────────────────────────────────────────────────────────────
# PERSON DESCRIPTORS
# ──────────────────────────────────────────────────────────────

PERSON_POSES: FrozenSet[str] = frozenset(
    {
        "sitting",
        "standing",
        "lying",
        "waving",
        "pointing",
        "raising",
    }
)

BODY_SIDES: FrozenSet[str] = frozenset({"left", "right"})

BODY_PARTS: FrozenSet[str] = frozenset({"arm", "hand", "leg", "foot"})

# color + garment as separate sets for the Matcher patterns
CLOTHING_COLORS: FrozenSet[str] = frozenset(
    {
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "white",
        "black",
        "gray",
        "grey",
        "brown",
        "purple",
        "pink",
    }
)

CLOTHING_ITEMS: FrozenSet[str] = frozenset(
    {
        "shirt",
        "coat",
        "jacket",
        "sweater",
        "t shirt",
        "blouse",
        "skirt",
        "pants",
        "jeans",
    }
)

# ──────────────────────────────────────────────────────────────
# PHYSICAL ATTRIBUTES (superlatives used in QUERY commands)
# ──────────────────────────────────────────────────────────────

SUPERLATIVE_MAP: Dict[str, str] = {
    "largest": "biggest",
    "biggest": "biggest",
    "greatest": "biggest",
    "smallest": "smallest",
    "tiniest": "smallest",
    "littlest": "smallest",
    "tallest": "tallest",
    "highest": "tallest",
    "shortest": "shortest",
    "lowest": "shortest",
    "heaviest": "heaviest",
    "lightest": "lightest",
    "oldest": "oldest",
    "eldest": "oldest",
    "youngest": "youngest",
    "widest": "widest",
    "narrowest": "narrowest",
    "thickest": "thickest",
    "thinnest": "thinnest",
    "fullest": "fullest",
    "emptiest": "emptiest",
}

# ──────────────────────────────────────────────────────────────
# QUERY ATTRIBUTE NOUNS
# ──────────────────────────────────────────────────────────────

PERCEPTUAL_ATTRIBUTES: FrozenSet[str] = frozenset(
    {
        "pose",
        "name",
        "gesture",
        "color",
        "colour",
        "age",
        "gender",
        "size",
        "shape",
        "weight",
        "height",
        "position",
        "status",
        "description",
    }
)

# ──────────────────────────────────────────────────────────────
# SOCIAL / INFO QUERIES the robot can answer from internal state
# ──────────────────────────────────────────────────────────────

SELF_INFO_TOKENS: FrozenSet[str] = frozenset(
    {
        "affiliation",
        "name",
        "country",
        "team",
        "time",
        "date",
        "today",
        "tomorrow",
        "week",
    }
)
