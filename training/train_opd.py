"""OPD entrypoint: load YAML config, build dataset (with curriculum filter),
load student + frozen teacher, hand to MoralSteerTrainer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.data_utils.roleplay_dataset import DEFAULT_TRAIN_JSON, build_dataset
from training.trainer.moralsteer_trainer import MoralSteerConfig, MoralSteerTrainer


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config for this OPD stage.")
    ap.add_argument("--override-output-dir", default=None)
    ap.add_argument("--override-student-model", default=None)
    ap.add_argument("--override-teacher-model", default=None)
    ap.add_argument("--limit-scenarios", type=int, default=None, help="Smoke option (raw json path only).")
    ap.add_argument("--max-rows", type=int, default=None, help="Cap dataset to N rows (tier1 path).")
    args = ap.parse_args()

    cfg_raw = _load_yaml(args.config)

    student_path = args.override_student_model or cfg_raw.pop("student_model")
    teacher_path = args.override_teacher_model or cfg_raw.pop("teacher_model", None)
    cfg_raw.pop("student_model", None)
    cfg_raw.pop("teacher_model", None)
    tier1_jsonl = cfg_raw.pop("tier1_jsonl", None)
    source_json = cfg_raw.pop("source_json", None)  # legacy: raw RolePlay_Villain.json
    m_max = cfg_raw.pop("m_max", None)
    oversample_l4 = int(cfg_raw.pop("oversample_l4", 1))

    if tier1_jsonl is not None:
        source, source_kind = tier1_jsonl, "tier1"
    elif source_json is not None:
        source, source_kind = source_json, "rolepaly_villain_json"
    else:
        source, source_kind = str(DEFAULT_TRAIN_JSON), "rolepaly_villain_json"

    if args.override_output_dir is not None:
        cfg_raw["output_dir"] = args.override_output_dir
    # Auto-derive logging_dir if not given, so tensorboard files land under the run.
    if not cfg_raw.get("logging_dir"):
        cfg_raw["logging_dir"] = str(Path(cfg_raw["output_dir"]) / "tb")

    # MoralSteerConfig inherits HF TrainingArguments fields, so all generic
    # training args go through here.
    cfg_raw["teacher_model_name_or_path"] = teacher_path
    sd_cfg = MoralSteerConfig(**cfg_raw)

    tok = AutoTokenizer.from_pretrained(student_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = build_dataset(
        source=source,
        source_kind=source_kind,
        m_max=m_max,
        oversample_l4=oversample_l4,
        limit_scenarios=args.limit_scenarios,
        max_rows=args.max_rows,
    )
    print(
        f"[opd] dataset size: {len(train_ds)}  source={source}  kind={source_kind}  "
        f"m_max={m_max} oversample_l4={oversample_l4}"
    )

    # Load student as text-only class (Qwen3_5ForCausalLM). Multimodal class
    # triggers a Qwen3.5 M-RoPE bug on text-only inputs (see train_sft_warmup.py).
    print(f"[opd] loading student (text-only class) from {student_path}")
    student_model = AutoModelForCausalLM.from_pretrained(
        student_path,
        torch_dtype="bfloat16",
        trust_remote_code=True,
    )

    trainer = MoralSteerTrainer(
        model=student_model,
        args=sd_cfg,
        train_dataset=train_ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(sd_cfg.output_dir)

    # Carry the multimodal processor configs forward stage-to-stage; without
    # these, vLLM 0.17 refuses to load the next OPD stage's student.
    import shutil
    src_root = Path(student_path)
    dst_root = Path(sd_cfg.output_dir)
    for aux in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja"):
        src, dst = src_root / aux, dst_root / aux
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
            print(f"[opd] copied {aux}")


if __name__ == "__main__":
    main()
