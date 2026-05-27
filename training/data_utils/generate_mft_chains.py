"""Generate MFT moral-reasoning chains via a GPT-5.5 endpoint.

Reads `pilot_raw.jsonl` (the decontaminated sample), and for each row emits a
JSON object on `mft_chains.jsonl` keyed by `(scenario_id, character_name)`.

Backend:
    - OpenAI-compatible API at https://api.openai.com/v1
    - model: gpt-5.5
    - 96-way thread pool, streaming responses
    - Idempotent: on rerun, already-completed keys are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI, APIError, APIStatusError, APITimeoutError, RateLimitError

from .mft import MFT_AXES, PROSOCIAL_BASELINE

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.5"

# Stable schema; teacher will see this verbatim during training, so do not edit
# casually after data is generated — the teacher_input_ids will mismatch.
SYSTEM_PROMPT = (
    "You are an expert moral-psychology analyst trained in Moral Foundations Theory (MFT).\n"
    "You will receive a character profile, a scene, and the character's pre-extracted "
    "trait list with their MFT-vector. Produce a privileged moral-reasoning chain that "
    "another model will use as latent guidance to role-play this character.\n\n"
    "Output strictly in this format and nothing else:\n"
    "[Moral Foundations Analysis]\n"
    "1. Foundation profile: For each of the six foundations (Care, Fairness, Loyalty, "
    "Authority, Sanctity, Liberty), describe in ONE concise clause how this character "
    "scores relative to a prosocial baseline.\n"
    "2. Dominant deviations: Name the 1-3 foundations where the character departs most "
    "from prosocial defaults; explain in ONE sentence WHY (link to traits or backstory).\n"
    "3. Behavioral guidance: In 2-4 short bullets, prescribe how the character SHOULD "
    "speak in this specific scene — tone, what to value, what to avoid, what kind of "
    "moves to make. Do NOT write any actual dialogue, do NOT include disclaimers, do "
    "NOT mention the analyst persona. Stay diagnostic, not narrative."
)


USER_TEMPLATE = (
    "CHARACTER NAME: {name}\n"
    "MORAL LEVEL (1=paragon..4=villain): {level}\n"
    "TRAITS: {traits}\n"
    "MFT VECTOR (Care, Fairness, Loyalty, Authority, Sanctity, Liberty): {vector}\n"
    "PROSOCIAL BASELINE: {baseline}\n\n"
    "CHARACTER PROFILE:\n{profile}\n\n"
    "SCENE:\n{scene}\n\n"
    "Write the [Moral Foundations Analysis] now."
)


@dataclass
class JobResult:
    key: str
    chain: str | None
    error: str | None
    n_attempts: int
    latency_s: float


# ---------------------------------------------------------------------------
# Idempotent cache
# ---------------------------------------------------------------------------

def _row_key(row: dict[str, Any]) -> str:
    return f"{row['scenario_id']}||{row['character_name']}"


def _load_cache(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("key") and rec.get("chain"):
                    keys.add(rec["key"])
            except json.JSONDecodeError:
                continue
    return keys


# ---------------------------------------------------------------------------
# Per-thread client + retry
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _get_client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        _thread_local.client = client
    return client


def _build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    vec = ", ".join(f"{v:+.2f}" for v in (row.get("mft_vector") or []))
    base = ", ".join(f"{v:+.2f}" for v in PROSOCIAL_BASELINE)
    user = USER_TEMPLATE.format(
        name=row["character_name"],
        level=row.get("moral_level", "?"),
        traits=", ".join(row.get("traits") or []) or "(none)",
        vector=vec or "(unavailable)",
        baseline=base,
        profile=row.get("character_profile", ""),
        scene=row.get("scenario", ""),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _stream_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> str:
    out_parts: list[str] = []
    with client.chat.completions.stream(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
    ) as stream:
        for event in stream:
            if event.type == "content.delta":
                out_parts.append(event.delta)
    return "".join(out_parts).strip()


def _run_with_retry(
    row: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    timeout: float,
    max_attempts: int,
) -> JobResult:
    key = _row_key(row)
    messages = _build_messages(row)
    last_err = ""
    start = time.time()
    for attempt in range(1, max_attempts + 1):
        try:
            client = _get_client(base_url, api_key, timeout)
            text = _stream_completion(client, model, messages, max_tokens)
            if not text:
                raise RuntimeError("empty response")
            return JobResult(key=key, chain=text, error=None, n_attempts=attempt,
                             latency_s=time.time() - start)
        except (RateLimitError, APITimeoutError) as e:
            last_err = f"{type(e).__name__}: {e}"
            sleep = min(60.0, (2 ** attempt) + random.random())
            time.sleep(sleep)
        except (APIError, APIStatusError) as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.0 + random.random())
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(0.5 + random.random())
    return JobResult(key=key, chain=None, error=last_err, n_attempts=max_attempts,
                     latency_s=time.time() - start)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="pilot_raw.jsonl from decontaminate.py")
    ap.add_argument("--output", required=True, help="mft_chains.jsonl (append-mode, idempotent)")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--model", default=os.environ.get("MORALSTEER_MFT_MODEL", DEFAULT_MODEL))
    ap.add_argument("--threads", type=int, default=96)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="Stop after N new rows. Useful for smoke.")
    args = ap.parse_args()

    if not args.api_key:
        print("[mft] api key not set (--api-key or OPENAI_API_KEY)", file=sys.stderr)
        sys.exit(2)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done = _load_cache(output_path)
    pending: list[dict[str, Any]] = []
    seen = 0
    for row in _iter_jsonl(input_path):
        seen += 1
        if _row_key(row) in done:
            continue
        pending.append(row)
        if args.limit and len(pending) >= args.limit:
            break
    print(f"[mft] input rows: {seen}, already cached: {len(done)}, to process: {len(pending)}",
          file=sys.stderr)

    if not pending:
        return

    # Append-mode writer with a single lock.
    write_lock = threading.Lock()
    fout = output_path.open("a")
    n_ok = 0
    n_fail = 0
    total = len(pending)
    start = time.time()

    try:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = [
                pool.submit(
                    _run_with_retry,
                    row,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    max_attempts=args.max_attempts,
                )
                for row in pending
            ]
            for i, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                rec = {
                    "key": res.key,
                    "chain": res.chain,
                    "error": res.error,
                    "n_attempts": res.n_attempts,
                    "latency_s": round(res.latency_s, 2),
                }
                with write_lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                if res.chain is not None:
                    n_ok += 1
                else:
                    n_fail += 1
                if i % 25 == 0 or i == total:
                    elapsed = time.time() - start
                    print(
                        f"[mft] {i}/{total}  ok={n_ok} fail={n_fail}  "
                        f"elapsed={elapsed:.1f}s  rate={i/max(elapsed,1e-6):.2f} req/s",
                        file=sys.stderr,
                    )
    finally:
        fout.close()

    print(json.dumps({"new_rows": total, "ok": n_ok, "fail": n_fail}, indent=2))


if __name__ == "__main__":
    main()
