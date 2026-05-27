"""SFT warmup before OPD (Enhanced warmup, proposal Section 3.4).

We don't have CoSER nor the teacher-generated L3-L4 outputs yet, so this script
currently warms up on the Moral RolePlay official `dialogues` field, treating
each (prior_dialogues, next_utterance) pair from the target character as an SFT
example. This gives the student the basic "stay in character, do not refuse"
format the OPD stage depends on. Once CoSER + teacher L3-L4 samples are ready,
extend `build_sft_corpus` to mix them in.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from training.data_utils.roleplay_dataset import (
    DEFAULT_TRAIN_JSON,
    SYSTEM_STUDENT_TMPL,
    USER_TMPL,
    _format_dialogue_history,
)


def build_sft_corpus(json_path: Path) -> Dataset:
    raw = json.loads(json_path.read_text())
    rows: list[dict] = []
    for scene in raw:
        scenario = scene.get("scenario", "")
        topic = scene.get("topic", "")
        book = scene.get("book", "")
        profiles = scene.get("character_profiles") or {}
        dialogues = scene.get("dialogues") or []
        # Index turns by absolute position.
        for i, turn in enumerate(dialogues):
            name = turn.get("character")
            message = (turn.get("message") or "").strip()
            if not name or not message or name not in profiles:
                continue
            history = dialogues[:i]
            system_s = SYSTEM_STUDENT_TMPL.format(name=name, book=book, profile=profiles[name])
            user_msg = USER_TMPL.format(
                scenario=scenario, topic=topic, dialogue=_format_dialogue_history(history), name=name
            )
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": system_s},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": message},
                    ]
                }
            )
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Base model path or HF id.")
    ap.add_argument("--source", default=str(DEFAULT_TRAIN_JSON))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--limit-scenarios", type=int, default=None)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--per-device-bsz", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--report-to", default="tensorboard", help="Comma-separated: tensorboard,wandb,none")
    ap.add_argument("--logging-dir", default=None, help="Tensorboard log dir; defaults to <output-dir>/tb")
    args = ap.parse_args()

    src = Path(args.source)
    ds = build_sft_corpus(src)
    if args.limit_scenarios is not None:
        ds = ds.select(range(min(args.limit_scenarios, len(ds))))
    print(f"[sft_warmup] corpus size: {len(ds)}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load as text-only class. We previously tried AutoModelForImageTextToText
    # for vLLM 0.17 compatibility, but vLLM colocate is unworkable on 4xH200
    # under FSDP (bad_alloc on KV pool, see configs/opd_stage1.yaml), so we
    # use the transformers backend everywhere. Text-only class avoids the
    # Qwen3.5 multimodal M-RoPE forward path which crashes under text inputs
    # with "Sizes of tensors must match except in dimension 3" in
    # apply_rotary_pos_emb.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="bfloat16" if args.bf16 else "auto",
        trust_remote_code=True,
    )

    logging_dir = args.logging_dir or str(Path(args.output_dir) / "tb")
    report_to = [s.strip() for s in args.report_to.split(",") if s.strip()]

    cfg = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_length=args.max_seq_length,
        logging_steps=10,
        save_strategy="epoch",   # per-epoch ckpt; orchestrator prunes checkpoint-N after stage ends
        save_total_limit=1,
        bf16=args.bf16,
        report_to=report_to,
        logging_dir=logging_dir,
        gradient_checkpointing=True,
        dataset_text_field=None,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(args.output_dir)

    # HF Trainer.save_model() only serializes the `processing_class` (tokenizer
    # here). For Qwen3.5 the multimodal model expects an image+video processor
    # config too — vLLM 0.17 refuses to load a Qwen3.5 ckpt without
    # preprocessor_config.json + video_preprocessor_config.json. Copy them from
    # the base model so the SFT ckpt is a complete drop-in.
    import shutil
    src_root = Path(args.model)
    dst_root = Path(args.output_dir)
    for aux in ("preprocessor_config.json", "video_preprocessor_config.json"):
        src, dst = src_root / aux, dst_root / aux
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
            print(f"[sft_warmup] copied {aux}")

    print(f"[sft_warmup] saved warmup checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
