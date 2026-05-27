"""Decontaminate Moral RolePlay train set against the official test set,
then stratify-sample the surviving (scenario, character) rows.

Two filters (CoSER book-level filter is a no-op for now since we don't load CoSER):
    1. character-level: drop rows whose `character_name` appears in any test scenario's
       key_characters.
    2. scenario-level: drop rows whose `scenario` text shares 5-gram Jaccard > 0.7 with
       any test scenario.

Then perform stratified sampling by moral_level (L1..L4), with replacement only when
a stratum has fewer surviving rows than the per-stratum quota.

Outputs:
    {out_dir}/pilot_raw.jsonl              # the sampled (scenario, character) rows
    {out_dir}/decontamination_report.json  # counts, removal reasons, seeded RNG state
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .mft import build_profile

DEFAULT_TRAIN_JSON = Path(
    "./data/benchmarks/moral_roleplay/RolePlay_Villain.json"
)
DEFAULT_TEST_JSON = Path(
    "./data/benchmarks/moral_roleplay/RolePlay_Villain_test.json"
)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


def _ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    tokens = _TOKEN_RE.findall(text.lower())
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0


def _test_signatures(test_path: Path) -> tuple[set[str], list[set[tuple[str, ...]]]]:
    """Collect character-name set and a list of 5-gram fingerprints over test scenarios."""
    test = json.loads(test_path.read_text())
    chars: set[str] = set()
    fps: list[set[tuple[str, ...]]] = []
    for scene in test:
        for kc in scene.get("key_characters") or []:
            name = kc.get("name")
            if name:
                chars.add(name)
        fp = _ngrams(scene.get("scenario", ""))
        if fp:
            fps.append(fp)
    return chars, fps


def _scenario_max_jaccard(scene_text: str, test_fps: list[set[tuple[str, ...]]]) -> float:
    fp = _ngrams(scene_text)
    if not fp:
        return 0.0
    return max((_jaccard(fp, tfp) for tfp in test_fps), default=0.0)


def _format_dialogue(turns: list[dict[str, str]] | None, max_turns: int = 6) -> list[dict[str, str]]:
    if not turns:
        return []
    return [
        {"character": t.get("character", "?"), "message": (t.get("message") or "").strip()}
        for t in turns[-max_turns:]
    ]


def _gold_utterance(scene: dict[str, Any], target_name: str) -> str | None:
    """Return the first message by target_name in this scenario's dialogues (the SFT target)."""
    for t in scene.get("dialogues") or []:
        if t.get("character") == target_name:
            return (t.get("message") or "").strip() or None
    return None


def _expand_rows(
    train_path: Path,
    test_chars: set[str],
    test_fps: list[set[tuple[str, ...]]],
    jaccard_threshold: float,
) -> Iterable[dict[str, Any]]:
    train = json.loads(train_path.read_text())
    for scene in train:
        scene_text = scene.get("scenario", "")
        # Score scenario-level Jaccard once per scene.
        max_jacc = _scenario_max_jaccard(scene_text, test_fps)
        book = scene.get("book", "")
        i_p = scene.get("i_p")
        i_c = scene.get("i_c")
        scenario_id = f"{book}-{i_p}-{i_c}"
        topic = scene.get("topic", "")
        profiles = scene.get("character_profiles") or {}
        dialogue_history = _format_dialogue(scene.get("dialogues") or [])

        for kc in scene.get("key_characters") or []:
            name = kc.get("name")
            if not name:
                continue
            mc = kc.get("morality_classification") or {}
            level = mc.get("level")
            traits = list(mc.get("key_traits") or [])
            profile_text = profiles.get(name) or kc.get("thought") or ""

            status = "passed"
            if name in test_chars:
                status = "exact_char"
            elif max_jacc > jaccard_threshold:
                status = "jaccard_scene"

            mp = build_profile(traits, level=level)
            yield {
                "scenario_id": scenario_id,
                "book": book,
                "i_p": i_p if i_p is not None else -1,
                "i_c": i_c if i_c is not None else -1,
                "scenario": scene_text,
                "topic": topic,
                "dialogue_history": dialogue_history,
                "character_name": name,
                "character_profile": profile_text,
                "traits": traits,
                "moral_level": level if level is not None else -1,
                "mft_vector": list(mp.mft_vector),
                "m_c": mp.m_c,
                "gold_utterance": _gold_utterance(scene, name),
                "decontamination": {
                    "status": status,
                    "scene_jaccard_max": round(max_jacc, 4),
                    "test_char_match": name if status == "exact_char" else None,
                },
            }


def _stratified_sample(
    passed: list[dict[str, Any]],
    quotas: dict[int, int],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample `quotas[level]` rows from each moral_level stratum.

    If a stratum has fewer rows than its quota, sample without replacement from
    everything available (we do NOT replicate here — the trainer's
    `oversample_l4` knob is the right place for replication).
    """
    by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in passed:
        by_level[int(row["moral_level"])].append(row)
    chosen: list[dict[str, Any]] = []
    realized = {}
    for level, q in quotas.items():
        pool = by_level.get(level, [])
        rng.shuffle(pool)
        k = min(q, len(pool))
        chosen.extend(pool[:k])
        realized[level] = {"requested": q, "available": len(pool), "taken": k}
    rng.shuffle(chosen)
    return chosen, realized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(DEFAULT_TRAIN_JSON))
    ap.add_argument("--test", default=str(DEFAULT_TEST_JSON))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jaccard-threshold", type=float, default=0.7)
    ap.add_argument(
        "--quota",
        default="1:1500,2:1500,3:1000,4:1000",
        help="Per-moral_level sample quota, format 'level:n,level:n,...'. "
             "Default L1=1500 L2=1500 L3=1000 L4=1000 -> 5000.",
    )
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quotas = {int(k): int(v) for kv in args.quota.split(",") for k, v in [kv.split(":")]}

    test_chars, test_fps = _test_signatures(Path(args.test))
    print(f"[decontaminate] test chars: {len(test_chars)}, test scene fingerprints: {len(test_fps)}",
          file=sys.stderr)

    rows = list(
        _expand_rows(
            Path(args.train),
            test_chars=test_chars,
            test_fps=test_fps,
            jaccard_threshold=args.jaccard_threshold,
        )
    )
    print(f"[decontaminate] raw rows expanded: {len(rows)}", file=sys.stderr)

    status_counts = Counter(r["decontamination"]["status"] for r in rows)
    passed = [r for r in rows if r["decontamination"]["status"] == "passed"]

    rng = random.Random(args.seed)
    sampled, realized = _stratified_sample(passed, quotas, rng)

    pilot_path = out_dir / "pilot_raw.jsonl"
    with pilot_path.open("w") as f:
        for row in sampled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "raw_rows": len(rows),
        "passed_rows": len(passed),
        "status_counts": dict(status_counts),
        "quotas": quotas,
        "realized_by_level": realized,
        "sampled_rows": len(sampled),
        "seed": args.seed,
        "jaccard_threshold": args.jaccard_threshold,
    }
    (out_dir / "decontamination_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
