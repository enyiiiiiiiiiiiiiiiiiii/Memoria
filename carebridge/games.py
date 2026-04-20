from __future__ import annotations

import random
from typing import Any

from .config import ASSETS_DIR

EMOJI_TILES = ["🌻", "☕", "🌈", "🎵", "🍎", "🐱", "⭐", "🦋"]

# The files are copied from the uploaded Zip reference folder.  They map to
# the Naumi Singapore kopi guide order: 4 x 4 grid from left-to-right,
# top-to-bottom.
KOPI_DRINK_CARDS: list[dict[str, str]] = [
    {
        "name": "Kopi",
        "label": "Kopi",
        "alt_label": "Black coffee with milk and sugar",
        "image": "Screenshot 2026-04-18 182646.png",
    },
    {
        "name": "Kopi Kosong",
        "label": "Kopi Kosong",
        "alt_label": "Black coffee with milk and no sugar",
        "image": "Screenshot 2026-04-18 182651.png",
    },
    {
        "name": "Kopi Siew Dai",
        "label": "Kopi Siew Dai",
        "alt_label": "Black coffee with milk and less sugar",
        "image": "Screenshot 2026-04-18 185849.png",
    },
    {
        "name": "Kopi Gah Dai",
        "label": "Kopi Gah Dai",
        "alt_label": "Black coffee with milk and more sugar",
        "image": "Screenshot 2026-04-18 185855.png",
    },
    {
        "name": "Kopi Gau",
        "label": "Kopi Gau",
        "alt_label": "Strong black coffee with milk and sugar",
        "image": "Screenshot 2026-04-18 185903.png",
    },
    {
        "name": "Kopi Po",
        "label": "Kopi Po",
        "alt_label": "Weak coffee with milk and sugar",
        "image": "Screenshot 2026-04-18 185906.png",
    },
    {
        "name": "Kopi-C",
        "label": "Kopi-C",
        "alt_label": "Black coffee with evaporated milk and sugar",
        "image": "Screenshot 2026-04-18 185910.png",
    },
    {
        "name": "Kopi-C Kosong",
        "label": "Kopi-C Kosong",
        "alt_label": "Black coffee with evaporated milk and no sugar",
        "image": "Screenshot 2026-04-18 185912.png",
    },
    {
        "name": "Kopi-O",
        "label": "Kopi-O",
        "alt_label": "Americano with sugar",
        "image": "Screenshot 2026-04-18 185915.png",
    },
    {
        "name": "Kopi-O Kosong",
        "label": "Kopi-O Kosong",
        "alt_label": "Americano with no sugar",
        "image": "Screenshot 2026-04-18 185918.png",
    },
    {
        "name": "Kopi-O Siew Dai",
        "label": "Kopi-O Siew Dai",
        "alt_label": "Americano with less sugar",
        "image": "Screenshot 2026-04-18 185920.png",
    },
    {
        "name": "Kopi-O Gau",
        "label": "Kopi-O Gau",
        "alt_label": "Strong americano with sugar",
        "image": "Screenshot 2026-04-18 185923.png",
    },
    {
        "name": "Kopi-O Po",
        "label": "Kopi-O Po",
        "alt_label": "Weak americano with sugar",
        "image": "Screenshot 2026-04-18 185926.png",
    },
    {
        "name": "Kopi-O Kosong Di Lo",
        "label": "Kopi-O Kosong Di Lo",
        "alt_label": "Long black with no sugar",
        "image": "Screenshot 2026-04-18 185928.png",
    },
    {
        "name": "Kopi Peng",
        "label": "Kopi Peng",
        "alt_label": "Iced black coffee with milk and sugar",
        "image": "Screenshot 2026-04-18 185930.png",
    },
    {
        "name": "Kopi-O Peng",
        "label": "Kopi-O Peng",
        "alt_label": "Iced americano with sugar",
        "image": "Screenshot 2026-04-18 185932.png",
    },
]

# Backwards-compatible names imported by older app code.
KOPI_INGREDIENTS = ["coffee", "tea", "condensed milk", "evaporated milk", "sugar"]
KOPI_RECIPES = {item["name"].lower(): set() for item in KOPI_DRINK_CARDS}


def build_memory_board(seed: int | None = None) -> list[str]:
    rng = random.Random(seed)
    tiles = EMOJI_TILES[:8]
    board = tiles + tiles
    rng.shuffle(board)
    return board


def compute_game_points(score: int, moves: int, duration_seconds: int) -> int:
    efficiency_bonus = max(0, 30 - moves)
    speed_bonus = max(0, 180 - duration_seconds) // 12
    return max(10, int(score + efficiency_bonus + speed_bonus))


def reward_for_points(total_points: int) -> dict[str, Any] | None:
    rewards = [
        {"threshold": 600, "voucher_name": "$5 Community Cafe", "partner_name": "Local Kopitiam"},
        {"threshold": 1000, "voucher_name": "$8 Pharmacy Voucher", "partner_name": "Neighbourhood Pharmacy"},
        {"threshold": 1500, "voucher_name": "$12 Hawker Treat", "partner_name": "Heartland Hawker"},
    ]
    eligible = [item for item in rewards if total_points >= item["threshold"]]
    return eligible[-1] if eligible else None


def kopi_image_path(card: dict[str, str]) -> str:
    return str(ASSETS_DIR / "kopi" / card["image"])


def new_kopi_round(seed: int | None = None) -> dict[str, Any]:
    rng = random.Random(seed)
    target = rng.choice(KOPI_DRINK_CARDS)
    distractors = [item for item in KOPI_DRINK_CARDS if item["name"] != target["name"]]
    options = rng.sample(distractors, 3) + [target]
    rng.shuffle(options)
    return {
        "target_drink": target["name"],
        "target_label": target["label"],
        "target_alt_label": target["alt_label"],
        "image_path": kopi_image_path(target),
        "options": [{"name": item["name"], "label": item["label"]} for item in options],
        "result": None,
    }


def evaluate_kopi_guess(target_drink: str, selected_drink: str) -> dict[str, Any]:
    correct = selected_drink == target_drink
    score = 100 if correct else 20
    return {
        "correct": correct,
        "score": score,
        "message": f"Correct — this is {target_drink}." if correct else f"Not quite right. This was {target_drink}.",
        "expected": target_drink,
        "selected": selected_drink,
    }


def evaluate_kopi_mix(target_drink: str, selected: list[str]) -> dict[str, Any]:
    """Legacy wrapper kept so older imports do not break."""
    selected_name = selected[0] if selected else ""
    return evaluate_kopi_guess(target_drink, selected_name)
