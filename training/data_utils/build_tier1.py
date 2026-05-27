"""Merge `pilot_raw.jsonl` and `mft_chains.jsonl` into a single `tier1.jsonl`.

Tier-1 is the canonical training seed used by every method (MoralSteer main,
OPSD no-MFT, SFT-MFT, DPO-MFT, Naive FT). The schema is documented in the
data-design memo. Rows whose MFT chain is missing or empty are dropped to
keep all derived datasets coherent (with a count in the report).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _key(row: dict[str, Any]) -> str:
    return f"{row['scenario_id']}||{row['character_name']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="pilot_raw.jsonl")
    ap.add_argument("--chains", required=True, help="mft_chains.jsonl")
    ap.add_argument("--out", required=True, help="tier1.jsonl path")
    ap.add_argument("--report", default=None, help="optional report JSON path")
    args = ap.parse_args()

    raw_path = Path(args.raw)
    chains_path = Path(args.chains)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chain_by_key: dict[str, str] = {}
    chain_errors = 0
    for rec in _iter_jsonl(chains_path):
        key = rec.get("key")
        chain = rec.get("chain")
        if key and isinstance(chain, str) and chain.strip():
            chain_by_key[key] = chain.strip()
        else:
            chain_errors += 1

    print(f"[tier1] chain cache: {len(chain_by_key)} usable, {chain_errors} unusable", file=sys.stderr)

    n_total = 0
    n_emit = 0
    missing_chain = 0
    by_level_emit: dict[int, int] = {}
    with out_path.open("w") as fout:
        for row in _iter_jsonl(raw_path):
            n_total += 1
            key = _key(row)
            chain = chain_by_key.get(key)
            if not chain:
                missing_chain += 1
                continue
            row_out = dict(row)
            row_out["mft_reasoning_chain"] = chain
            row_out["teacher_rollout"] = None
            fout.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            n_emit += 1
            lv = int(row_out.get("moral_level", -1))
            by_level_emit[lv] = by_level_emit.get(lv, 0) + 1

    report = {
        "raw_rows": n_total,
        "emitted_rows": n_emit,
        "missing_chain": missing_chain,
        "chain_cache_size": len(chain_by_key),
        "by_level_emit": dict(sorted(by_level_emit.items())),
    }
    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
