#!/usr/bin/env python3
"""Pre-training diagnostic gate for Fourier-Kronecker V3.

This script intentionally contains no Transformer and no training loop.  Its
job is to reject a bad positional basis before GPU time is spent on language
model training.

V3 is an Anchored Band-Limited Complex-Phase basis represented in real numbers:

    F(p)[2k]   = cos(omega[k] * p)
    F(p)[2k+1] = sin(omega[k] * p)

The cos/sin pair is exactly the real/imaginary decomposition of Euler's phase
exp(i * omega[k] * p).  Keeping the two components as ordinary real channels
means the Kronecker space is still 256 * 32 = 8192 dimensions and W_proj keeps
the same [8192, d_model] shape.

Frequency design
----------------
* omega[0] is a global anchor with wavelength 256 positions.
* omega[1:16] are geometrically spaced from 0.2 to 0.9*pi radians/position.
  Their wavelengths range from about 31.4 positions down to about 2.22.

The single slow phase supplies global absolute location over the diagnostic
window.  Fifteen medium/high phases prevent neighboring positions from becoming
nearly collinear.  Frequencies are not restricted to integer DFT bins, so there
is no forced common period at position 256.  The codec accepts every integer p;
there is no position table and therefore no hard word-length cutoff.

Run:

    python fourier_kronecker_v3_diagnostics.py

The script prints a human-readable report, writes a JSON result, and exits with
status 1 if any pre-training gate fails.  Use --no-strict to inspect a failing
candidate without returning a failing process status.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class V3Config:
    """All quantities required to reconstruct the deterministic V3 codec."""

    d_char: int = 256
    d_pos: int = 32
    diagnostic_positions: int = 256

    # One slow complex phase completes a revolution over 256 byte positions.
    global_wavelength: float = 256.0

    # Fifteen local angular frequencies cover both exact-order and neighborhood
    # information.  Values are in radians per byte position.
    local_omega_min: float = 0.2
    local_omega_max: float = 0.9 * math.pi

    z_norm_eps: float = 1.0e-5
    rank_relative_tolerance: float = 1.0e-6

    # Absolute acceptance thresholds selected before language-model training.
    minimum_effective_rank: float = 28.0
    maximum_off_diagonal_cosine: float = 0.60
    maximum_cat_act_cosine: float = 0.75
    minimum_typo_similarity_vs_v1_margin: float = -0.01
    minimum_cat_act_improvement_vs_v2: float = 0.20

    @property
    def n_complex_frequencies(self) -> int:
        return self.d_pos // 2

    @property
    def codec_dim(self) -> int:
        return self.d_char * self.d_pos


def geometric_values(start: float, stop: float, count: int) -> torch.Tensor:
    """Geometric spacing implemented explicitly in log space.

    Linear spacing in frequency would devote too many channels to the upper
    band.  Linear spacing in wavelength would do the opposite.  Log spacing
    gives every octave comparable capacity.
    """

    if not (start > 0.0 and stop > start and count > 0):
        raise ValueError("Need 0 < start < stop and count > 0")
    return torch.exp(
        torch.linspace(math.log(start), math.log(stop), count, dtype=torch.float64)
    )


def v3_angular_frequencies(config: V3Config) -> torch.Tensor:
    """Construct omega[0:16] without consuming additional dimensions."""

    if config.d_pos <= 0 or config.d_pos % 2:
        raise ValueError("d_pos must be a positive even number")
    if config.d_char != 256:
        raise ValueError("UTF-8 byte one-hot requires d_char=256")

    global_omega = torch.tensor(
        [2.0 * math.pi / config.global_wavelength], dtype=torch.float64
    )
    local_count = config.n_complex_frequencies - 1
    local_omegas = geometric_values(
        config.local_omega_min, config.local_omega_max, local_count
    )
    return torch.cat((global_omega, local_omegas))


def phase_basis(positions: torch.Tensor, omegas: torch.Tensor) -> torch.Tensor:
    """Evaluate real Euler features for one or many positions.

    Broadcasting math:
      positions[:, None] has shape [P, 1]
      omegas[None, :]   has shape [1, K]
      phase             has shape [P, K]

    Interleaving cos and sin produces [P, 2K] = [P, 32].  Every row has exact
    squared norm K because cos(theta)^2 + sin(theta)^2 = 1 for each frequency.
    """

    positions = positions.to(dtype=torch.float64).reshape(-1, 1)
    omegas = omegas.to(dtype=torch.float64).reshape(1, -1)
    phase = positions * omegas
    output = torch.empty(
        (positions.shape[0], 2 * omegas.shape[1]), dtype=torch.float64
    )
    output[:, 0::2] = torch.cos(phase)
    output[:, 1::2] = torch.sin(phase)
    return output


def original_v2_basis(positions: torch.Tensor, d_pos: int = 32) -> torch.Tensor:
    """Reconstruct the failed V2 basis for an apples-to-apples diagnostic.

    V2 used denominator tau ** (2k / (d_pos/2)).  This reaches approximately
    3.16e7 at k=15, causing many phases to remain almost constant for p<=255.
    """

    tau = 10_000.0
    half = d_pos // 2
    k = torch.arange(half, dtype=torch.float64)
    denominators = tau ** (2.0 * k / float(half))
    # phase_basis expects angular frequencies omega, and p/denominator is
    # identical to omega*p with omega=1/denominator.
    return phase_basis(positions, 1.0 / denominators)


def population_z_normalize(vector: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-token normalization across all D=8192 pre-projection coordinates."""

    centered = vector - vector.mean()
    variance = centered.square().mean()  # population variance: unbiased=False
    return centered * torch.rsqrt(variance + eps)


class FourierKroneckerV3Codec:
    """Deterministic UTF-8 byte x anchored-band-phase Kronecker codec."""

    def __init__(self, config: V3Config) -> None:
        self.config = config
        self.omegas = v3_angular_frequencies(config)

    def encode_bytes(self, value: bytes) -> torch.Tensor:
        if not value:
            raise ValueError("Cannot encode an empty byte string")

        length = len(value)
        positions = torch.arange(length, dtype=torch.float64)
        waves = phase_basis(positions, self.omegas)  # [L, 32]

        # Reshaping the 8192-vector as [256,32] exposes the Kronecker structure.
        # At position p, c_{b_p} is a 256-D one-hot, so the outer product
        # c_{b_p} x F(p) writes F(p) into exactly row b_p.  index_add_ sums wave
        # interference when a byte occurs more than once.
        byte_rows = torch.zeros(
            (self.config.d_char, self.config.d_pos), dtype=torch.float64
        )
        byte_indices = torch.tensor(list(value), dtype=torch.long)
        byte_rows.index_add_(0, byte_indices, waves)

        # The full L participates: no min(L,32), truncation, padding, or position
        # lookup table appears anywhere in this computation.
        codec = byte_rows.reshape(-1) / math.sqrt(float(length))
        return population_z_normalize(codec, self.config.z_norm_eps)

    def encode_text(self, text: str) -> torch.Tensor:
        return self.encode_bytes(text.encode("utf-8"))


def encode_original_v2(text: str, config: V3Config) -> torch.Tensor:
    """Original V2 codec, retained only as a rejected-basis comparator."""

    value = text.encode("utf-8")
    waves = original_v2_basis(torch.arange(len(value)), config.d_pos)
    rows = torch.zeros((config.d_char, config.d_pos), dtype=torch.float64)
    rows.index_add_(0, torch.tensor(list(value), dtype=torch.long), waves)
    codec = rows.reshape(-1) / math.sqrt(float(len(value)))
    return population_z_normalize(codec, config.z_norm_eps)


def encode_v1(text: str, config: V3Config) -> torch.Tensor:
    """Truncated one-hot Kronecker V1 comparator for string-level metrics."""

    value = text.encode("utf-8")[: config.d_pos]
    if not value:
        raise ValueError("Cannot encode an empty string")
    codec = torch.zeros(config.codec_dim, dtype=torch.float64)
    byte_indices = torch.tensor(list(value), dtype=torch.long)
    positions = torch.arange(len(value), dtype=torch.long)
    codec[byte_indices * config.d_pos + positions] = 1.0
    codec /= math.sqrt(float(len(value)))
    return population_z_normalize(codec, config.z_norm_eps)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left[None, :], right[None, :]).item())


def effective_rank(singular_values: torch.Tensor) -> float:
    """Entropy effective rank from normalized singular-value energy.

    If all 32 directions carry equal energy, effective rank is 32.  If one
    almost-constant direction dominates, the value approaches 1.  This catches
    V2's collapse more honestly than algebraic rank alone.
    """

    energy = singular_values.square()
    probability = energy / energy.sum()
    nonzero = probability > 0
    entropy = -(probability[nonzero] * torch.log(probability[nonzero])).sum()
    return float(torch.exp(entropy).item())


def basis_diagnostics(
    basis: torch.Tensor, relative_rank_tolerance: float
) -> Dict[str, object]:
    """Measure conditioning and the worst positional near-collision."""

    row_norms = torch.linalg.vector_norm(basis, dim=1, keepdim=True)
    normalized = basis / row_norms.clamp_min(torch.finfo(basis.dtype).eps)
    gram = normalized @ normalized.T

    # Remove self-similarity before searching for the most similar distinct
    # positions.  We report signed and absolute maxima because anticorrelation
    # can also matter for destructive interference.
    count = gram.shape[0]
    diagonal = torch.eye(count, dtype=torch.bool)
    signed_search = gram.masked_fill(diagonal, -torch.inf)
    absolute_search = gram.abs().masked_fill(diagonal, -torch.inf)

    signed_flat = int(torch.argmax(signed_search).item())
    absolute_flat = int(torch.argmax(absolute_search).item())
    signed_pair = divmod(signed_flat, count)
    absolute_pair = divmod(absolute_flat, count)

    singular_values = torch.linalg.svdvals(basis)
    numerical_rank = int(
        (singular_values > relative_rank_tolerance * singular_values[0]).sum().item()
    )
    condition_number = float((singular_values[0] / singular_values[-1]).item())

    return {
        "matrix_shape": list(basis.shape),
        "effective_rank": effective_rank(singular_values),
        "numerical_rank": numerical_rank,
        "condition_number": condition_number,
        "maximum_off_diagonal_cosine": float(signed_search.flatten()[signed_flat].item()),
        "maximum_cosine_position_pair": list(signed_pair),
        "maximum_absolute_off_diagonal_cosine": float(
            absolute_search.flatten()[absolute_flat].item()
        ),
        "maximum_absolute_cosine_position_pair": list(absolute_pair),
        "adjacent_position_cosine_0_1": float(gram[0, 1].item()),
        "minimum_row_norm": float(row_norms.min().item()),
        "maximum_row_norm": float(row_norms.max().item()),
        "singular_values": [float(value) for value in singular_values],
    }


def string_pair_diagnostics(config: V3Config) -> Dict[str, object]:
    """Measure robustness and order discrimination without cherry-picking one score."""

    v3 = FourierKroneckerV3Codec(config)
    encoders = {
        "v1": lambda text: encode_v1(text, config),
        "v2": lambda text: encode_original_v2(text, config),
        "v3": v3.encode_text,
    }
    pairs = {
        # Same bytes in different order: lower similarity is better because a
        # bag-of-characters representation would map them identically.
        "exact_order_cat_act": ("cat", "act"),
        "exact_order_listen_silent": ("listen", "silent"),
        # A one-character spelling variant: higher similarity is generally
        # useful, provided exact-order pairs remain distinguishable.
        "typo_separate_seperate": ("separate", "seperate"),
    }

    results: Dict[str, object] = {}
    for metric_name, (left, right) in pairs.items():
        values = {
            arm: cosine(encoder(left), encoder(right))
            for arm, encoder in encoders.items()
        }
        results[metric_name] = {
            "left": left,
            "right": right,
            "cosine": values,
            "discrimination_one_minus_cosine": {
                arm: 1.0 - value for arm, value in values.items()
            },
        }
    return results


def evaluate_gates(
    config: V3Config,
    v2_basis: Mapping[str, object],
    v3_basis: Mapping[str, object],
    strings: Mapping[str, object],
) -> Dict[str, bool]:
    """Apply thresholds fixed in V3Config and return every decision separately."""

    cat = strings["exact_order_cat_act"]  # type: ignore[index]
    typo = strings["typo_separate_seperate"]  # type: ignore[index]
    cat_cosines = cat["cosine"]  # type: ignore[index]
    typo_cosines = typo["cosine"]  # type: ignore[index]

    return {
        "effective_rank": float(v3_basis["effective_rank"])
        >= config.minimum_effective_rank,
        "maximum_off_diagonal_cosine": float(
            v3_basis["maximum_off_diagonal_cosine"]
        )
        <= config.maximum_off_diagonal_cosine,
        "cat_vs_act_absolute_discrimination": float(cat_cosines["v3"])
        <= config.maximum_cat_act_cosine,
        "cat_vs_act_improvement_over_v2": (
            float(cat_cosines["v2"]) - float(cat_cosines["v3"])
        )
        >= config.minimum_cat_act_improvement_vs_v2,
        "typo_similarity_preserved_vs_v1": float(typo_cosines["v3"])
        >= (
            float(typo_cosines["v1"])
            + config.minimum_typo_similarity_vs_v1_margin
        ),
        "effective_rank_improves_over_v2": float(v3_basis["effective_rank"])
        > float(v2_basis["effective_rank"]),
    }


def print_report(report: Mapping[str, object]) -> None:
    v2 = report["basis_diagnostics"]["v2"]  # type: ignore[index]
    v3 = report["basis_diagnostics"]["v3"]  # type: ignore[index]
    strings = report["string_diagnostics"]  # type: ignore[assignment]
    gates = report["gates"]  # type: ignore[assignment]

    print("=" * 80)
    print("FOURIER-KRONECKER V3 PRE-TRAINING DIAGNOSTIC GATE")
    print("=" * 80)
    print("Angular frequencies (radians/position):")
    print("  " + ", ".join(f"{value:.6f}" for value in report["omegas"]))  # type: ignore[arg-type]

    print("\nPOSITION BASIS OVER p=0..255")
    for name, metrics in (("Rejected V2", v2), ("Candidate V3", v3)):
        print(
            f"{name:12s} effective_rank={metrics['effective_rank']:.4f}/32  "
            f"numerical_rank={metrics['numerical_rank']:2d}/32  "
            f"max_offdiag_cos={metrics['maximum_off_diagonal_cosine']:.6f}  "
            f"cos(F0,F1)={metrics['adjacent_position_cosine_0_1']:.6f}  "
            f"condition={metrics['condition_number']:.3e}"
        )

    print("\nSTRING-LEVEL CODEC COSINES")
    for metric_name, metric in strings.items():
        cosines = metric["cosine"]
        print(
            f"{metric_name:30s} V1={cosines['v1']:.6f}  "
            f"V2={cosines['v2']:.6f}  V3={cosines['v3']:.6f}"
        )

    print("\nGATE DECISIONS")
    for name, passed in gates.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\nOVERALL: {'PASS' if report['overall_pass'] else 'FAIL'}")


def run_diagnostics(config: V3Config) -> Dict[str, object]:
    positions = torch.arange(config.diagnostic_positions, dtype=torch.float64)
    omegas = v3_angular_frequencies(config)
    v2_matrix = original_v2_basis(positions, config.d_pos)
    v3_matrix = phase_basis(positions, omegas)

    v2_metrics = basis_diagnostics(v2_matrix, config.rank_relative_tolerance)
    v3_metrics = basis_diagnostics(v3_matrix, config.rank_relative_tolerance)
    string_metrics = string_pair_diagnostics(config)
    gates = evaluate_gates(config, v2_metrics, v3_metrics, string_metrics)

    # Constructive no-cutoff sanity check.  A 1024-byte token still produces one
    # fixed [8192] vector, and every byte contributes before summation.
    long_vector = FourierKroneckerV3Codec(config).encode_bytes(b"V3" * 512)
    no_cutoff = {
        "input_bytes": 1024,
        "output_shape": list(long_vector.shape),
        "all_finite": bool(torch.isfinite(long_vector).all().item()),
    }
    gates["no_hard_cutoff_sanity"] = (
        no_cutoff["output_shape"] == [config.codec_dim]
        and no_cutoff["all_finite"]
    )

    return {
        "codec": "Anchored Band-Limited Complex-Phase Fourier-Kronecker V3",
        "config": asdict(config),
        "omegas": [float(value) for value in omegas],
        "wavelengths": [float(2.0 * math.pi / value) for value in omegas],
        "basis_diagnostics": {"v2": v2_metrics, "v3": v3_metrics},
        "string_diagnostics": string_metrics,
        "no_hard_cutoff_sanity": no_cutoff,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "fourier_kronecker_v3_diagnostics.json",
        help="JSON report path.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Return exit status 0 even if a diagnostic gate fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = V3Config()
    report = run_diagnostics(config)
    print_report(report)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nSaved JSON report: {output_path}")

    if not report["overall_pass"] and not args.no_strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
