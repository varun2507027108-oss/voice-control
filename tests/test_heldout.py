"""Held-out intent evaluation test suite to verify semantic generalization and fail-closed safety."""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from voice_controller.engine_intent import get_intent_router

HELD_OUT_POS = [
    ("crank it up", "volume_up"),
    ("it's too loud in here", "volume_down"),
    ("my screen is way too bright", "brightness_down"),
    ("start the next episode", "media_next"),
    ("put the internet on", "open_browser"),
    ("make everything louder", "volume_up"),
    ("secure my pc", "lock_workstation"),
]

HELD_OUT_NEG = [
    "turn on the lights",        # surface match -> brightness_up
    "open the fridge",           # surface match -> open_*
    "lock the door",             # surface match -> lock_workstation
    "play a game of solitaire",  # surface match -> media_play_pause
    "lower the price",
    "what's the weather tomorrow",
    "add milk to the shopping list",
    "how tall is mount everest",
]


def test_held_out_positives():
    """Verify routing accuracy on held-out positive paraphrases, asserting semantic Laya routing."""
    router = get_intent_router()

    laya_source_count = 0
    failures = []

    print("\n=== HELD-OUT POSITIVE EVALUATION ===")
    for utterance, expected_action in HELD_OUT_POS:
        res = router.predict(utterance)
        action = res.get("action")
        conf = res.get("confidence", 0.0)
        source = res.get("source")
        latency_ms = res.get("latency_ms", 0.0)

        if source == "laya":
            laya_source_count += 1

        print(f"[{source:>12}] '{utterance}' -> {action} (conf={conf*100:.1f}%, lat={latency_ms:.1f}ms)")

        if action != expected_action:
            failures.append(f"'{utterance}': expected '{expected_action}', got '{action}' ({source}, conf={conf:.2f})")

    assert not failures, f"Held-out positive failures:\n" + "\n".join(failures)

    # Condition 1: At least several positives must route through Laya semantic path
    print(f"\nPositives routed through Laya semantic path: {laya_source_count}/{len(HELD_OUT_POS)}")
    assert laya_source_count >= 3, f"Expected at least 3 positives to route via 'laya', got {laya_source_count}"


def test_held_out_negatives():
    """Verify adversarial negatives fail closed to 'unrecognized', printing top-2 and margin on failure."""
    router = get_intent_router()

    failures = []
    print("\n=== HELD-OUT ADVERSARIAL NEGATIVE EVALUATION ===")
    for utterance in HELD_OUT_NEG:
        res = router.predict(utterance)
        action = res.get("action")
        conf = res.get("confidence", 0.0)
        source = res.get("source")
        margin = res.get("margin", 0.0)
        top2 = res.get("top2", [])

        print(f"[{source:>12}] '{utterance}' -> {action} (conf={conf*100:.1f}%, margin={margin:.2f})")

        if action != "unrecognized":
            top2_str = ", ".join([f"{k}={v:.3f}" for k, v in top2]) if top2 else "none"
            failures.append(
                f"LEAKAGE: '{utterance}' routed to '{action}' ({source}, conf={conf:.2f}, margin={margin:.2f}, top-2: [{top2_str}])"
            )

    assert not failures, f"Adversarial negative failures:\n" + "\n".join(failures)
    print(f"All {len(HELD_OUT_NEG)} adversarial negatives successfully failed closed to 'unrecognized'.")


if __name__ == "__main__":
    test_held_out_positives()
    test_held_out_negatives()
    print("\n>>> ALL HELD-OUT TESTS PASSED! <<<")
