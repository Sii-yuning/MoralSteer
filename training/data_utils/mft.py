"""Moral Foundations Theory (MFT) primitives for MoralSteer.

Maps character traits to 6-dim MFT vectors, defines a prosocial baseline,
computes the moral-complexity scalar M(c), and renders a placeholder MFT
reasoning chain used as the teacher privileged context until the real
GPT-4o-mini chains are produced offline.

MFT axes (signed, in [-1, +1]; positive = prosocial pole):
    0: Care / Harm
    1: Fairness / Cheating
    2: Loyalty / Betrayal
    3: Authority / Subversion
    4: Sanctity / Degradation
    5: Liberty / Oppression
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

MFT_AXES = (
    "Care",
    "Fairness",
    "Loyalty",
    "Authority",
    "Sanctity",
    "Liberty",
)

PROSOCIAL_BASELINE = (0.8, 0.8, 0.6, 0.4, 0.4, 0.5)

# Placeholder trait -> 6-d MFT polarity. Values in [-1, 1].
# Sources: GPT-4 seed mapping + manual review of Moral RolePlay top-K traits.
# This is the "occupied seat" of the proposal Section 4.3 mapping table;
# must be replaced by 3-annotator Likert means + Krippendorff alpha report.
TRAIT_TO_MFT: dict[str, tuple[float, float, float, float, float, float]] = {
    # Prosocial / heroic
    "Kind":           (+0.9, +0.6, +0.4, +0.2, +0.3, +0.4),
    "Compassionate":  (+0.9, +0.5, +0.5, +0.1, +0.4, +0.3),
    "Loyal":          (+0.4, +0.3, +0.9, +0.4, +0.3, +0.2),
    "Brave":          (+0.5, +0.3, +0.4, +0.3, +0.3, +0.6),
    "Upright":        (+0.6, +0.8, +0.5, +0.5, +0.5, +0.4),
    "Wise":           (+0.5, +0.6, +0.4, +0.5, +0.4, +0.5),
    "Resilient":      (+0.3, +0.2, +0.3, +0.2, +0.2, +0.5),
    "Rational":       (+0.2, +0.5, +0.2, +0.3, +0.2, +0.4),
    "Cautious":       (+0.3, +0.3, +0.3, +0.4, +0.3, +0.2),
    "Patient":        (+0.4, +0.5, +0.4, +0.3, +0.3, +0.3),
    # Mixed / flawed-but-good
    "Stubborn":       (-0.1, +0.1, +0.2, -0.3, +0.0, +0.3),
    "Impulsive":      (-0.2, -0.2, +0.0, -0.4, -0.2, +0.5),
    "Melancholic":    (-0.1, +0.0, +0.0, +0.0, -0.1, +0.0),
    "Emotional":      (+0.1, -0.1, +0.2, -0.2, +0.0, +0.1),
    "Ambitious":      (-0.2, -0.3, +0.1, +0.3, +0.0, +0.4),
    "Worldly":        (+0.0, -0.1, +0.1, +0.2, -0.2, +0.3),
    "Arrogant":       (-0.4, -0.5, -0.2, +0.5, -0.2, -0.1),
    # Egoist (L3)
    "Selfish":        (-0.6, -0.6, -0.4, +0.0, -0.2, +0.4),
    "Greedy":         (-0.5, -0.7, -0.3, +0.0, -0.3, +0.2),
    "Cunning":        (-0.3, -0.5, -0.2, +0.1, -0.2, +0.3),
    "Manipulative":   (-0.7, -0.8, -0.5, -0.1, -0.3, -0.2),
    "Deceitful":      (-0.5, -0.9, -0.4, -0.2, -0.2, -0.1),
    "Vindictive":     (-0.7, -0.5, -0.3, +0.0, -0.2, +0.0),
    # Villain (L4)
    "Cruel":          (-0.9, -0.7, -0.4, +0.0, -0.5, -0.3),
    "Malicious":      (-0.9, -0.8, -0.5, -0.2, -0.5, -0.4),
    "Violent":        (-0.9, -0.6, -0.3, -0.1, -0.5, -0.5),
    "Sadistic":       (-1.0, -0.8, -0.4, +0.0, -0.7, -0.4),
    "Treacherous":    (-0.5, -0.7, -0.9, -0.4, -0.3, -0.2),
    "Ruthless":       (-0.8, -0.6, -0.3, +0.2, -0.4, +0.0),
    "Domineering":    (-0.5, -0.5, -0.2, +0.6, -0.2, -0.7),
    "Tyrannical":     (-0.8, -0.7, -0.3, +0.7, -0.3, -0.9),
}

# Used when a trait is not in the table; a neutral vector contributes 0 to M(c).
NEUTRAL_VECTOR = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def trait_vector(trait: str) -> tuple[float, float, float, float, float, float]:
    return TRAIT_TO_MFT.get(trait, NEUTRAL_VECTOR)


def character_mft_vector(traits: Iterable[str]) -> tuple[float, ...]:
    """Mean MFT vector over a character's key_traits (unknown traits drop to neutral)."""
    vecs = [trait_vector(t) for t in traits]
    if not vecs:
        return NEUTRAL_VECTOR
    n = len(vecs)
    return tuple(sum(v[i] for v in vecs) / n for i in range(6))


def moral_complexity(traits: Iterable[str]) -> float:
    """M(c) = (1/6) * sum_i |mu_i(c) - mu_i^prosocial|.

    Bounded in [0, ~1.8] given values in [-1, +1] and a prosocial baseline of
    roughly +0.5; in practice L1 paragons sit near 0.2, L4 villains near 1.0.
    """
    mu = character_mft_vector(traits)
    return sum(abs(mu[i] - PROSOCIAL_BASELINE[i]) for i in range(6)) / 6.0


def alpha_from_m(m_c: float, tau: float = 0.45, k: float = 6.0) -> float:
    """alpha(c) = sigmoid(k * (M(c) - tau)).

    Following paper Eq. (5): L = (1-alpha)*RKL + alpha*FKL.
    High M(c) -> high alpha -> FKL (coverage, prevents prosocial collapse);
    Low  M(c) -> low  alpha -> RKL (mode-seeking, prevents over-darkening).
    """
    return 1.0 / (1.0 + math.exp(-k * (m_c - tau)))


def lambda_from_m(m_c: float, beta: float = 0.8, tau: float = 0.55, k: float = 6.0) -> float:
    """lambda(c) = 1 - beta * sigmoid(k * (M(c) - tau)) in [1-beta, 1].

    Probability of using student rollout (vs. teacher trajectory) for sample c.
    """
    return 1.0 - beta * (1.0 / (1.0 + math.exp(-k * (m_c - tau))))


# --- Placeholder MFT reasoning chain renderer ---------------------------------

_REASONING_TEMPLATE = (
    "[Moral Foundations Analysis]\n"
    "Character traits: {trait_list}.\n"
    "MFT profile (Care, Fairness, Loyalty, Authority, Sanctity, Liberty) = "
    "({c:+.2f}, {f:+.2f}, {l:+.2f}, {a:+.2f}, {s:+.2f}, {lib:+.2f}).\n"
    "Salient deviations from a prosocial baseline:\n{deviation_lines}\n"
    "Behavioral guidance for in-character speech: {behavior_hint}\n"
)


def _behavior_hint(mu: tuple[float, ...]) -> str:
    hints: list[str] = []
    care_delta = mu[0] - PROSOCIAL_BASELINE[0]
    fair_delta = mu[1] - PROSOCIAL_BASELINE[1]
    auth_delta = mu[3] - PROSOCIAL_BASELINE[3]
    if care_delta < -0.4:
        hints.append("express low empathy and tolerate harm to others")
    elif care_delta > 0.2:
        hints.append("show concern for others' suffering")
    if fair_delta < -0.4:
        hints.append("rationalize unfair or deceptive moves")
    if auth_delta > 0.3:
        hints.append("invoke status, hierarchy, or command")
    elif auth_delta < -0.3:
        hints.append("undermine institutional authority")
    if not hints:
        hints.append("stay close to ordinary moral defaults")
    return "; ".join(hints) + "."


def render_reasoning_chain(traits: Iterable[str]) -> str:
    """Render a placeholder MFT reasoning chain from the character's traits.

    To be replaced by GPT-4o-mini–generated chains (proposal Section 4.1, ~50k
    samples). Keep the schema stable so downstream training does not change.
    """
    traits = list(traits)
    mu = character_mft_vector(traits)
    deviations = []
    for i, axis in enumerate(MFT_AXES):
        delta = mu[i] - PROSOCIAL_BASELINE[i]
        if abs(delta) >= 0.3:
            direction = "below" if delta < 0 else "above"
            deviations.append(f"  - {axis}: {delta:+.2f} ({direction} prosocial baseline)")
    if not deviations:
        deviations.append("  - (no axis deviates by >=0.3 from baseline)")
    return _REASONING_TEMPLATE.format(
        trait_list=", ".join(traits) if traits else "(none reported)",
        c=mu[0], f=mu[1], l=mu[2], a=mu[3], s=mu[4], lib=mu[5],
        deviation_lines="\n".join(deviations),
        behavior_hint=_behavior_hint(mu),
    )


@dataclass(frozen=True)
class MoralProfile:
    traits: tuple[str, ...]
    level: int | None
    mft_vector: tuple[float, ...]
    m_c: float
    reasoning_chain: str


def build_profile(traits: Iterable[str], level: int | None = None) -> MoralProfile:
    traits = tuple(traits)
    mu = character_mft_vector(traits)
    return MoralProfile(
        traits=traits,
        level=level,
        mft_vector=mu,
        m_c=moral_complexity(traits),
        reasoning_chain=render_reasoning_chain(traits),
    )


if __name__ == "__main__":
    # quick smoke
    for traits in (
        ["Kind", "Brave", "Loyal"],
        ["Stubborn", "Impulsive", "Loyal"],
        ["Manipulative", "Cunning", "Selfish"],
        ["Cruel", "Sadistic", "Tyrannical"],
    ):
        p = build_profile(traits)
        print(f"traits={traits} M(c)={p.m_c:.3f} alpha={alpha_from_m(p.m_c):.3f} lambda={lambda_from_m(p.m_c):.3f}")
