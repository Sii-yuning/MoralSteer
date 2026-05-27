"""Build the OPD training dataset from a Tier-1 jsonl seed.

Each Tier-1 row already contains the decontaminated (scenario, character)
information plus an MFT reasoning chain. This module renders the trl-friendly
`prompt` / `privileged_context` chat messages and applies curriculum filters.

Schema emitted to the trainer:
    prompt              : list[{role, content}]  # student-visible (no MFT)
    privileged_context  : list[{role, content}]  # teacher-visible (with MFT chain)
    m_c                 : float
    moral_level         : int
    character_name      : str
    book, i_p, i_c
    traits              : list[str]
    teacher_completion  : str | None    (Layer-2 fallback target, populated later)

Legacy path: a caller may still pass the raw `RolePlay_Villain.json` via
`source_kind="rolepaly_villain_json"`. In that case the loader expands rows on
the fly and renders a placeholder MFT chain (training/data_utils/mft.py).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import Dataset

from .mft import build_profile

PROJECT_ROOT = Path(".")
DEFAULT_TIER1_JSONL = PROJECT_ROOT / "training/artifacts/data/pilot5k/tier1.jsonl"
DEFAULT_TRAIN_JSON = (
    PROJECT_ROOT / "data/benchmarks/moral_roleplay/RolePlay_Villain.json"
)
DEFAULT_TEST_JSON = (
    PROJECT_ROOT / "data/benchmarks/moral_roleplay/RolePlay_Villain_test.json"
)

SYSTEM_STUDENT_TMPL = (
    "You are role-playing as {name} from the work \"{book}\".\n"
    "Stay strictly in character. Never break the fourth wall. Do not add safety "
    "disclaimers, do not refuse the role, and do not insert moral commentary as "
    "the narrator. Speak only as {name} would speak in this scene.\n\n"
    "Character profile:\n{profile}"
)

SYSTEM_TEACHER_SUFFIX = (
    "\n\n[Privileged moral-foundations analysis — visible only to the teacher]\n{mft_chain}"
)

USER_TMPL = (
    "Scene:\n{scenario}\n\n"
    "Topic of this scene: {topic}\n\n"
    "Recent dialogue:\n{dialogue}\n\n"
    "Now speak as {name}. Respond with a single in-character utterance."
)


def _format_dialogue_history(history: list[dict[str, str]] | None, max_turns: int = 6) -> str:
    if not history:
        return "(no prior dialogue)"
    trimmed = history[-max_turns:]
    return "\n".join(f"{t.get('character','?')}: {(t.get('message') or '').strip()}" for t in trimmed)


def _render_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    name = row["character_name"]
    book = row.get("book", "")
    profile = row.get("character_profile", "")
    scenario = row.get("scenario", "")
    topic = row.get("topic", "")
    dialogue = _format_dialogue_history(row.get("dialogue_history"))

    chain = row.get("mft_reasoning_chain")
    if not chain:
        chain = build_profile(row.get("traits") or [], level=row.get("moral_level")).reasoning_chain

    system_s = SYSTEM_STUDENT_TMPL.format(name=name, book=book, profile=profile)
    system_t = system_s + SYSTEM_TEACHER_SUFFIX.format(mft_chain=chain)
    user_msg = USER_TMPL.format(scenario=scenario, topic=topic, dialogue=dialogue, name=name)
    return (
        [{"role": "system", "content": system_s}, {"role": "user", "content": user_msg}],
        [{"role": "system", "content": system_t}, {"role": "user", "content": user_msg}],
    )


def _row_to_example(row: dict[str, Any]) -> dict[str, Any]:
    prompt, privileged = _render_messages(row)
    return {
        "prompt": prompt,
        "privileged_context": privileged,
        "m_c": float(row.get("m_c", 0.0)),
        "moral_level": int(row.get("moral_level", -1)),
        "character_name": row["character_name"],
        "book": row.get("book", ""),
        "i_p": int(row.get("i_p", -1) if row.get("i_p") is not None else -1),
        "i_c": int(row.get("i_c", -1) if row.get("i_c") is not None else -1),
        "traits": list(row.get("traits") or []),
        "teacher_completion": row.get("teacher_rollout"),
    }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _iter_tier1(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_rolepaly_villain_json(path: Path, limit_scenarios: int | None) -> Iterable[dict[str, Any]]:
    """Legacy path: read raw RolePlay_Villain.json (used by smoke tests before
    Tier-1 exists) and synthesize the Tier-1-shaped row on the fly."""
    raw = json.loads(path.read_text())
    if limit_scenarios is not None:
        raw = raw[:limit_scenarios]
    for scene in raw:
        scenario = scene.get("scenario", "")
        topic = scene.get("topic", "")
        book = scene.get("book", "")
        i_p = scene.get("i_p")
        i_c = scene.get("i_c")
        scenario_id = f"{book}-{i_p}-{i_c}"
        profiles = scene.get("character_profiles") or {}
        history = [
            {"character": t.get("character"), "message": (t.get("message") or "").strip()}
            for t in (scene.get("dialogues") or [])
        ]
        for kc in scene.get("key_characters") or []:
            name = kc.get("name")
            if not name:
                continue
            mc = kc.get("morality_classification") or {}
            level = mc.get("level")
            traits = list(mc.get("key_traits") or [])
            mp = build_profile(traits, level=level)
            yield {
                "scenario_id": scenario_id,
                "book": book,
                "i_p": i_p, "i_c": i_c,
                "scenario": scenario,
                "topic": topic,
                "dialogue_history": history,
                "character_name": name,
                "character_profile": profiles.get(name) or kc.get("thought") or "",
                "traits": traits,
                "moral_level": level,
                "mft_vector": list(mp.mft_vector),
                "m_c": mp.m_c,
                "mft_reasoning_chain": mp.reasoning_chain,
                "teacher_rollout": None,
            }


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_dataset(
    source: str | Path | None = None,
    *,
    source_kind: str = "tier1",
    m_max: float | None = None,
    oversample_l4: int = 1,
    limit_scenarios: int | None = None,
    max_rows: int | None = None,
    json_path: str | Path | None = None,  # back-compat: alias for source
) -> Dataset:
    """Materialize the OPD dataset.

    Args:
        source: path to the data file. Defaults to DEFAULT_TIER1_JSONL.
        source_kind: 'tier1' (jsonl) or 'rolepaly_villain_json' (raw HuggingFace dump).
        m_max: drop rows with M(c) > m_max (curriculum stage filter).
        oversample_l4: replicate moral_level==4 rows this many times.
        limit_scenarios: stop after the first N scenarios (rolepaly_villain_json only).
        json_path: alias for `source` (kept for back-compat with earlier callers).
    """
    if json_path is not None and source is None:
        source = json_path
        # heuristic: a json file is the raw RolePlay_Villain dump
        if source_kind == "tier1" and str(source).endswith(".json"):
            source_kind = "rolepaly_villain_json"

    if source is None:
        source = DEFAULT_TIER1_JSONL
    source = Path(source)

    if source_kind == "tier1":
        seed_iter = _iter_tier1(source)
    elif source_kind == "rolepaly_villain_json":
        seed_iter = _iter_rolepaly_villain_json(source, limit_scenarios)
    else:
        raise ValueError(f"Unknown source_kind: {source_kind}")

    rows: list[dict[str, Any]] = []
    for seed in seed_iter:
        m_c = float(seed.get("m_c", 0.0))
        if m_max is not None and m_c > m_max:
            continue
        example = _row_to_example(seed)
        rows.append(example)
        if oversample_l4 > 1 and example["moral_level"] == 4:
            for _ in range(oversample_l4 - 1):
                rows.append(example)
        if max_rows is not None and len(rows) >= max_rows:
            break

    return Dataset.from_list(rows)


def _summarize(ds: Dataset) -> dict[str, Any]:
    by_level: dict[int, int] = {}
    m_sum, m_n = 0.0, 0
    m_min, m_max = float("inf"), float("-inf")
    for row in ds:
        lv = int(row["moral_level"])
        by_level[lv] = by_level.get(lv, 0) + 1
        mc = float(row["m_c"])
        m_sum += mc
        m_n += 1
        m_min, m_max = min(m_min, mc), max(m_max, mc)
    return {
        "n_rows": len(ds),
        "by_level": dict(sorted(by_level.items())),
        "m_c_mean": m_sum / max(m_n, 1),
        "m_c_min": m_min if m_n else None,
        "m_c_max": m_max if m_n else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(DEFAULT_TIER1_JSONL))
    ap.add_argument("--source-kind", choices=("tier1", "rolepaly_villain_json"), default="tier1")
    ap.add_argument("--m-max", type=float, default=None)
    ap.add_argument("--oversample-l4", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = build_dataset(
        source=args.source,
        source_kind=args.source_kind,
        m_max=args.m_max,
        oversample_l4=args.oversample_l4,
        limit_scenarios=args.limit,
    )
    print(json.dumps(_summarize(ds), indent=2))
    if args.out:
        ds.save_to_disk(args.out)
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
