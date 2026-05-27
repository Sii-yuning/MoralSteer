"""MoralSteerTrainer: on-policy self-distillation with privileged MFT teacher.

This trainer extends `trl.experimental.self_distillation.BaseSelfDistillationTrainer`
to implement the OPD framework in the MoralSteer proposal:

1. Student is conditioned on x_S = [character_profile; scene]; Teacher is the
   *same frozen weights* conditioned on x_T = x_S + MFT reasoning chain.
   Implemented by tokenizing each sample's `privileged_context` into
   `teacher_input_ids` so the mixin's existing loss computes the KL on shared
   `completion_ids` while the two sides see different prefixes.

2. The distillation_alpha (KL direction) is per-sample: alpha(c) = sigmoid(M(c)-tau).
   We override `_compute_self_distillation_loss` to compute FKL and RKL on the
   same student/teacher log-probs and mix by per-sample alpha.

3. lambda(c) rollout mixing (Layer 2). A quality check on each student rollout
   marks degenerate samples; for them, `self_distillation_mask = 0` so they do
   not contribute to the loss. (Pre-generated teacher trajectories can be
   plugged in later via `teacher_completion` column on the dataset; for now
   the failed rollouts are simply dropped from the loss.)

4. The policy-loss branch from the SDPO base is skipped: `_compute_loss`
   returns only the distillation term, weighted by `distillation_weight`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset, IterableDataset
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)

from trl.experimental.self_distillation.base_self_distillation_trainer import (
    BaseSelfDistillationTrainer,
)
from trl.experimental.self_distillation.self_distillation_config import (
    SelfDistillationConfig,
)
from trl.trainer.utils import pad, selective_log_softmax

from .quality import passes_quality


_VLLM_PATCHED = False


def _patch_vllm_for_qwen35() -> None:
    """Force vLLM-safe defaults for Qwen3.5 / Qwen3-MoE.

    trl.generation.vllm_generation.VLLMGeneration constructs vLLM's `LLM(...)`
    without `enforce_eager` or `disable_custom_all_reduce`. Qwen3.5's hybrid
    attention (linear_attn + sliding window) is not graph-capturable on the
    current vLLM 0.17 build — CUDA graph capture native-crashes inside
    `std::vector::reserve`. The official Qwen3.5 vLLM serve script always
    passes `--enforce-eager` and `--disable-custom-all-reduce`, so we replicate
    that here as setdefault kwargs on the LLM constructor.
    """
    global _VLLM_PATCHED
    if _VLLM_PATCHED:
        return
    try:
        from vllm import LLM
    except ImportError:
        return
    original_init = LLM.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("enforce_eager", True)
        kwargs.setdefault("disable_custom_all_reduce", True)
        return original_init(self, *args, **kwargs)

    LLM.__init__ = patched_init
    _VLLM_PATCHED = True


@dataclass
class MoralSteerConfig(SelfDistillationConfig):
    """Configuration for MoralSteerTrainer.

    Adds:
        teacher_model_name_or_path: separate frozen teacher checkpoint. If None,
            the student's initial weights are used as the teacher (only valid
            until the first optimizer step — not recommended in production).
        m_alpha_tau / m_alpha_k: sigmoid params for alpha(c) (RKL weight).
        m_lambda_beta / m_lambda_tau / m_lambda_k: sigmoid params for lambda(c).
        quality_min_chars / quality_max_repeat_ratio: rollout quality thresholds.
    """

    teacher_model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Path to frozen teacher checkpoint. Falls back to student weights if None."},
    )
    m_alpha_tau: float = field(default=0.45, metadata={"help": "Sigmoid threshold for alpha(c)."})
    m_alpha_k: float = field(default=6.0, metadata={"help": "Sigmoid steepness for alpha(c)."})
    m_lambda_beta: float = field(
        default=0.8,
        metadata={"help": "Upper-bound reduction of lambda(c) (max teacher-fallback probability)."},
    )
    m_lambda_tau: float = field(default=0.55, metadata={"help": "Sigmoid threshold for lambda(c)."})
    m_lambda_k: float = field(default=6.0, metadata={"help": "Sigmoid steepness for lambda(c)."})
    quality_min_chars: int = field(default=20, metadata={"help": "Min characters for a passing rollout."})
    quality_max_repeat_ratio: float = field(
        default=0.35,
        metadata={"help": "Max trigram repeat ratio before rollout is flagged degenerate."},
    )

    def __post_init__(self):
        super().__post_init__()
        if not (0.0 <= self.m_lambda_beta <= 1.0):
            raise ValueError("m_lambda_beta must be in [0, 1]")


def _alpha_of_m(m_c: torch.Tensor, tau: float, k: float) -> torch.Tensor:
    return torch.sigmoid(k * (m_c - tau))


def _lambda_of_m(m_c: torch.Tensor, tau: float, k: float, beta: float) -> torch.Tensor:
    return 1.0 - beta * torch.sigmoid(k * (m_c - tau))


class MoralSteerTrainer(BaseSelfDistillationTrainer):
    config_cls = MoralSteerConfig
    _name = "MoralSteer"
    _tag_names = ["trl", "moralsteer"]

    def __init__(
        self,
        model: str | PreTrainedModel | nn.Module,
        args: MoralSteerConfig,
        train_dataset: Dataset | IterableDataset,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config=None,
    ):
        # Force vLLM-safe defaults BEFORE trl's base trainer constructs vLLM.
        if getattr(args, "use_vllm", False):
            _patch_vllm_for_qwen35()

        # We don't use rewards. Pass a single dummy reward func so the base
        # class skips the RL critic path cleanly.
        def _dummy_reward(prompts, completions, completion_ids, **_):
            return [1.0] * len(prompts)

        super().__init__(
            model=model,
            reward_funcs=[_dummy_reward],
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

        # Load the frozen teacher (separate weights) if a path was provided.
        # Text-only class — see train_sft_warmup.py for why we don't use the
        # multimodal class.
        if args.teacher_model_name_or_path is not None:
            teacher = AutoModelForCausalLM.from_pretrained(
                args.teacher_model_name_or_path,
                **(args.model_init_kwargs or {}),
            )
            teacher.requires_grad_(False)
            teacher.eval()
            self.teacher_model = self._prepare_auxiliary_model_for_eval(teacher)

    # ------------------------------------------------------------------
    # FSDP-aware generation: when training under FSDP FULL_SHARD, the model's
    # parameters are flat-sharded 1-D tensors on each rank, so any forward path
    # that doesn't first `summon_full_params` (incl. transformers `.generate()`
    # via accelerate.unwrap_model_for_generation) raises
    #   RuntimeError: 'weight' must be 2-D
    # or AssertionError: Padding_idx must be within num_embeddings.
    # accelerate 1.13's unwrap_model_for_generation only auto-summons for
    # DeepSpeed ZeRO-3, not FSDP, so we wrap the parent's rollout method here.
    # ------------------------------------------------------------------
    def _generate_transformers(self, prompts):
        if getattr(self, "is_fsdp_enabled", False):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            with FSDP.summon_full_params(self.model, writeback=False, recurse=True):
                return super()._generate_transformers(prompts)
        return super()._generate_transformers(prompts)

    # ------------------------------------------------------------------
    # Signature columns: keep prompt/privileged_context + our extras.
    # ------------------------------------------------------------------
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt",
                "privileged_context",
                "m_c",
                "moral_level",
                "character_name",
                "book",
                "i_p",
                "i_c",
                "traits",
                "teacher_completion",
            ]

    # ------------------------------------------------------------------
    # Build teacher inputs and per-sample diagnostics on top of the base
    # generation pipeline.
    # ------------------------------------------------------------------
    def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        output = super()._generate_and_score_completions(inputs)

        device = self.accelerator.device
        completion_ids = output["completion_ids"]
        completion_mask = output["completion_mask"]

        # 1) Tokenize privileged contexts to teacher prompt ids (left-padded).
        privileged = [ex["privileged_context"] for ex in inputs]
        from trl.data_utils import maybe_apply_chat_template

        privileged_text = [
            maybe_apply_chat_template({"prompt": p}, self.processing_class, **self.chat_template_kwargs)["prompt"]
            for p in privileged
        ]
        teacher_prompt = self.processing_class(
            text=privileged_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        teacher_prompt = {k: v.to(device) for k, v in teacher_prompt.items()}
        teacher_input_ids = torch.cat([teacher_prompt["input_ids"], completion_ids], dim=1)
        teacher_attention_mask = torch.cat(
            [teacher_prompt["attention_mask"].long(), completion_mask], dim=1
        )

        # 2) Quality-check rollouts + sample-level lambda(c) mask.
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        m_c = torch.tensor(
            [float(ex.get("m_c", 0.0)) for ex in inputs], dtype=torch.float32, device=device
        )
        lam = _lambda_of_m(m_c, self.args.m_lambda_tau, self.args.m_lambda_k, self.args.m_lambda_beta)

        passes: list[bool] = []
        reasons: dict[str, int] = {}
        for txt in completions_text:
            ok, reason = passes_quality(
                txt,
                min_chars=self.args.quality_min_chars,
                max_repeat_ratio=self.args.quality_max_repeat_ratio,
            )
            passes.append(ok)
            if not ok:
                reasons[reason] = reasons.get(reason, 0) + 1
        passes_t = torch.tensor(passes, dtype=torch.float32, device=device)

        # Bernoulli draw with prob lambda(c): keep student rollout. For failed
        # rollouts we force-drop (mask=0). Surviving samples contribute to the
        # distillation loss; the failed ones are skipped for this step.
        rand = torch.rand_like(lam)
        keep_student = (rand < lam).float() * passes_t
        # NOTE: pre-generated teacher trajectories would replace the dropped
        # samples in-place. Until that pipeline exists, failed samples become
        # zero-contribution via the self_distillation_mask.
        self_distillation_mask = keep_student

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["moralsteer/rollout_pass_rate"].append(passes_t.mean().item())
        self._metrics[mode]["moralsteer/keep_student_rate"].append(keep_student.mean().item())
        self._metrics[mode]["moralsteer/lambda_mean"].append(lam.mean().item())
        self._metrics[mode]["moralsteer/m_c_mean"].append(m_c.mean().item())
        for reason, n in reasons.items():
            self._metrics[mode][f"moralsteer/rollout_drop/{reason}"].append(n / max(len(inputs), 1))

        output["teacher_input_ids"] = teacher_input_ids
        output["teacher_attention_mask"] = teacher_attention_mask
        output["self_distillation_mask"] = self_distillation_mask
        output["m_c"] = m_c
        return output

    # ------------------------------------------------------------------
    # Loss: per-sample alpha(c) mix of FKL + RKL. We compute the same
    # student/teacher logits the parent mixin would, then mix divergences
    # rather than using a single global alpha.
    # ------------------------------------------------------------------
    def _compute_self_distillation_loss(self, model, inputs: dict[str, Any]) -> torch.Tensor:
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        logits_to_keep = completion_ids.size(1)

        sd_mask = inputs.get("self_distillation_mask")
        if sd_mask is not None:
            response_mask = completion_mask * sd_mask.unsqueeze(1)
        else:
            response_mask = completion_mask

        mode = "train" if model.training else "eval"
        if response_mask.sum() == 0:
            self._log_self_distillation_metric(mode, "distillation_loss", 0.0)
            return torch.tensor(0.0, device=completion_ids.device, requires_grad=True)

        # ----- Student forward -----
        student_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        student_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        student_inputs = {
            "input_ids": student_input_ids,
            "attention_mask": student_attention_mask,
            "use_cache": False,
        }
        if "logits_to_keep" in self.model_kwarg_keys:
            student_inputs["logits_to_keep"] = logits_to_keep + 1
        student_logits = model(**student_inputs).logits[:, :-1, :][:, -logits_to_keep:, :] / self.temperature

        # ----- Teacher forward -----
        teacher_input_ids = inputs["teacher_input_ids"]
        teacher_attention_mask = inputs["teacher_attention_mask"]
        teacher_inputs = {
            "input_ids": teacher_input_ids,
            "attention_mask": teacher_attention_mask,
            "use_cache": False,
        }
        if "logits_to_keep" in self.model_kwarg_keys:
            teacher_inputs["logits_to_keep"] = logits_to_keep + 1
        teacher_model = self._get_teacher_model_for_self_distillation(model)
        with torch.no_grad(), self._get_teacher_context_for_self_distillation(model):
            teacher_logits = teacher_model(**teacher_inputs).logits[:, :-1, :][:, -logits_to_keep:, :] / self.temperature

        # ----- per-sample alpha(c) -----
        m_c = inputs.get("m_c")
        if m_c is None:
            alpha = torch.full(
                (student_logits.size(0),),
                float(self.args.distillation_alpha),
                device=student_logits.device,
                dtype=student_logits.dtype,
            )
        else:
            alpha = _alpha_of_m(m_c.to(student_logits.dtype), self.args.m_alpha_tau, self.args.m_alpha_k)

        use_topk = (
            self.args.distillation_topk is not None
            and (self.args.full_logit_distillation or self._allow_topk_without_full_logit_distillation())
        )
        if use_topk:
            student_lse = torch.logsumexp(student_logits, dim=-1, keepdim=True)
            topk_student_logits, topk_indices = torch.topk(
                student_logits, k=self.args.distillation_topk, dim=-1
            )
            s_lp = topk_student_logits - student_lse
            teacher_lse = torch.logsumexp(teacher_logits, dim=-1, keepdim=True)
            t_lp = torch.gather(teacher_logits, dim=-1, index=topk_indices) - teacher_lse
            if self.args.distillation_add_tail:
                s_lp = self._add_tail(s_lp)
                t_lp = self._add_tail(t_lp)
            else:
                s_lp = self._renorm_topk_log_probs(s_lp)
                t_lp = self._renorm_topk_log_probs(t_lp)
        elif self.args.full_logit_distillation:
            s_lp = F.log_softmax(student_logits, dim=-1)
            t_lp = F.log_softmax(teacher_logits, dim=-1)
        else:
            # Token-level surrogate (alpha is constrained to 1.0 = RKL by trl's
            # mixin). We follow the same convention here: just call the base
            # token-level path and skip per-sample alpha (M(c) becomes ignored).
            s_lse = torch.logsumexp(student_logits, dim=-1, keepdim=True)
            t_lse = torch.logsumexp(teacher_logits, dim=-1, keepdim=True)
            idx = completion_ids.unsqueeze(-1)
            s_logps = (torch.gather(student_logits, dim=-1, index=idx) - s_lse).squeeze(-1)
            t_logps = (torch.gather(teacher_logits, dim=-1, index=idx) - t_lse).squeeze(-1)
            per_token_loss = self._compute_token_level_distillation_loss(s_logps, t_logps)
            return self._finalize_loss(per_token_loss, response_mask, completion_ids, student_logits, inputs, mode)

        # alpha is per-sample [B]; broadcast to [B, T, V] via [B, 1, 1].
        alpha_b = alpha.view(-1, 1, 1).clamp(min=1e-6, max=1.0 - 1e-6)
        # FKL = KL(t || s) ; RKL = KL(s || t). F.kl_div(input, target, log_target=True)
        # treats `input` as log-Q (the "from") and target as log-P (the "to").
        kl_fkl = F.kl_div(s_lp, t_lp, reduction="none", log_target=True).sum(-1)  # [B,T]
        kl_rkl = F.kl_div(t_lp, s_lp, reduction="none", log_target=True).sum(-1)
        # Per-sample mixture following paper Eq. (5):
        #   L_t = (1 - alpha(c)) * RKL + alpha(c) * FKL
        # High M(c) -> high alpha -> FKL (coverage, prevents prosocial collapse);
        # Low  M(c) -> low  alpha -> RKL (mode-seeking, prevents over-darkening).
        alpha_bt = alpha.view(-1, 1)
        per_token_loss = (1.0 - alpha_bt) * kl_rkl + alpha_bt * kl_fkl
        return self._finalize_loss(per_token_loss, response_mask, completion_ids, student_logits, inputs, mode)

    def _finalize_loss(
        self,
        per_token_loss: torch.Tensor,
        response_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        student_logits: torch.Tensor,
        inputs: dict[str, Any],
        mode: str,
    ) -> torch.Tensor:
        # Layer 3: per-token IS clipping (reuses trl mixin facility).
        if self.args.distillation_is_clip is not None:
            old_log_probs = inputs.get("old_per_token_logps")
            if old_log_probs is not None:
                with torch.no_grad():
                    s_lse = torch.logsumexp(student_logits, dim=-1, keepdim=True)
                    idx = completion_ids.unsqueeze(-1)
                    s_logps = (torch.gather(student_logits, dim=-1, index=idx) - s_lse).squeeze(-1)
                per_token_loss = self._apply_importance_sampling_clipping(
                    per_token_loss, s_logps, old_log_probs, self.args.distillation_is_clip
                )

        loss = self._aggregate_self_distillation_loss(per_token_loss, response_mask)
        mean_distill = (per_token_loss * response_mask).sum() / response_mask.sum().clamp(min=1.0)
        self._log_self_distillation_metric(
            mode, "distillation_loss", self.accelerator.gather(mean_distill).mean().item()
        )
        return loss

    # ------------------------------------------------------------------
    # Skip the policy-loss branch from the SDPO base: only distill loss.
    # ------------------------------------------------------------------
    def _compute_loss(self, model, inputs) -> torch.Tensor:
        if self.args.distillation_weight <= 0.0:
            raise ValueError("MoralSteer requires distillation_weight > 0; no policy term is used.")
        accumulation_scale = self.current_gradient_accumulation_steps if self.model.training else 1.0
        sd_loss = self._compute_self_distillation_loss(model, inputs) / accumulation_scale
        return self.args.distillation_weight * sd_loss
