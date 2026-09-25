"""Create plots and aggregate uncertainty, cost, and memory analyses."""
from __future__ import annotations
import json
import math
from pathlib import Path
import statistics

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
GIB = 1024 ** 3
ORIGINAL = [
    ("Baseline b32", "baseline_precise_b32"),
    ("Midpoint b32", "midpoint_precise_b32"),
    ("Midpoint b100", "midpoint_precise_b100"),
    ("Euler b32", "euler_precise_b32"),
    ("Euler b102", "euler_precise_b102"),
]
SEED2 = {
    "baseline": "baseline_precise_b32_seed20260920",
    "midpoint": "midpoint_precise_b32_seed20260920",
    "euler": "euler_precise_b32_seed20260920",
}


def font(size, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def plot_training_curves(series):
    width, height = 1530, 884
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 120, 80, width - 55, height - 170
    y_min, y_max = 3.2, 9.4
    colors = ["#172c45", "#247ba0", "#70c1b3", "#ef8354", "#b24c63", "#745296", "#d18b00"]

    def px(tokens):
        return left + (tokens / 50_000_000) * (right - left)

    def py(loss):
        return bottom - ((loss - y_min) / (y_max - y_min)) * (bottom - top)

    for value in range(0, 51, 10):
        x = px(value * 1_000_000)
        draw.line((x, top, x, bottom), fill="#dde4ea", width=1)
        draw.text((x - 14, bottom + 12), str(value), font=font(22), fill="#334155")
    for loss in (4, 5, 6, 7, 8, 9):
        y = py(loss)
        draw.line((left, y, right, y), fill="#dde4ea", width=1)
        draw.text((65, y - 13), str(loss), font=font(22), fill="#334155")
    draw.line((left, top, left, bottom), fill="#334155", width=2)
    draw.line((left, bottom, right, bottom), fill="#334155", width=2)
    draw.text((width // 2 - 180, 20), "Validation loss during training", font=font(34, True), fill="#142c45")
    draw.text((width // 2 - 180, bottom + 46), "Supervised training tokens (millions)", font=font(25), fill="#334155")
    draw.text((left, top - 38), "Fixed-probe loss", font=font(22), fill="#334155")

    for index, (label, points) in enumerate(series):
        color = colors[index]
        pixels = [(px(token), py(loss)) for token, loss in points]
        draw.line(pixels, fill=color, width=4)
        for x, y in pixels:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        legend_x = left + (index % 3) * 430
        legend_y = bottom + 92 + (index // 3) * 38
        draw.line((legend_x, legend_y + 10, legend_x + 38, legend_y + 10), fill=color, width=5)
        draw.text((legend_x + 48, legend_y - 4), label, font=font(20), fill="#243044")
    image.save(ROOT / "training_curves.png")


def plot_seed_errorbars(seed_stats):
    width, height = 1120, 544
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 120, 80, width - 60, height - 100
    variants = ["baseline", "midpoint", "euler"]
    all_values = [v for variant in variants for v in seed_stats[variant]["validation_loss"]["values"]]
    y_min = min(all_values) - .06
    y_max = max(all_values) + .06

    def py(loss):
        return bottom - ((loss - y_min) / (y_max - y_min)) * (bottom - top)

    for i in range(5):
        value = y_min + i * (y_max - y_min) / 4
        y = py(value)
        draw.line((left, y, right, y), fill="#dde4ea", width=1)
        draw.text((35, y - 13), f"{value:.3f}", font=font(20), fill="#334155")
    draw.line((left, top, left, bottom), fill="#334155", width=2)
    draw.line((left, bottom, right, bottom), fill="#334155", width=2)
    draw.text((220, 18), "Two-seed loss variability (mean ± sample SD)", font=font(30, True), fill="#142c45")
    for index, variant in enumerate(variants):
        x = left + (index + .5) * (right - left) / 3
        stats = seed_stats[variant]["validation_loss"]
        mean, sd = stats["mean"], stats["sample_sd"]
        y_top, y_bottom, y_mean = py(mean + sd), py(mean - sd), py(mean)
        draw.line((x, y_top, x, y_bottom), fill="#ef8354", width=5)
        draw.line((x - 18, y_top, x + 18, y_top), fill="#ef8354", width=5)
        draw.line((x - 18, y_bottom, x + 18, y_bottom), fill="#ef8354", width=5)
        draw.ellipse((x - 8, y_mean - 8, x + 8, y_mean + 8), fill="#172c45")
        for dx, value in zip((-16, 16), stats["values"]):
            y = py(value)
            draw.ellipse((x + dx - 6, y - 6, x + dx + 6, y + 6), fill="#247ba0")
        draw.text((x - 48, bottom + 20), variant.title(), font=font(22), fill="#334155")
    image.save(ROOT / "seed_loss_errorbars.png")


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def fixed_curve(name):
    initial = read(f"{name}/initial_probe.json")
    points = [(0, initial["loss"])]
    for line in (ROOT / name / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event") == "validation" and event.get("scope") == "fixed_probe":
            points.append((event["consumed_tokens"], event["loss"]))
    by_token = {}
    for token, loss in points:
        by_token[token] = loss
    return sorted(by_token.items())


def mean_sd(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "values": values}


def main():
    tuning = read("lr_tuning.json")
    tuned = []
    for variant, data in tuning["variants"].items():
        name = data.get("full_run")
        if not name:
            peak = data["selected"]["peak_lr"]
            name = (f'{variant}_precise_b{data["batch"]}' if peak == 6e-4 else
                    f'{variant}_precise_b{data["batch"]}_lr{int(round(peak*1_000_000))}_tuned')
        tuned.append((f'{variant.title()} tuned b{data["batch"]}', name))

    original_names = {name for _, name in ORIGINAL}
    tuned_unique = [(label, name) for label, name in tuned if name not in original_names]
    curve_series = [(label, fixed_curve(name)) for label, name in ORIGINAL + tuned_unique]
    plot_training_curves(curve_series)

    seed_stats = {}
    for variant in ("baseline", "midpoint", "euler"):
        first = read(f'{variant}_precise_b32/result.json')
        second = read(f'{SEED2[variant]}/result.json')
        seed_stats[variant] = {
            "seeds": [20260919, 20260920],
            "validation_loss": mean_sd([first["final_validation"]["loss"], second["final_validation"]["loss"]]),
            "target_tokens_per_second": mean_sd([first["target_tokens_per_second"], second["target_tokens_per_second"]]),
            "peak_allocated_gib": mean_sd([first["peak_allocated_bytes"] / GIB, second["peak_allocated_bytes"] / GIB]),
        }

    plot_seed_errorbars(seed_stats)

    base = read("baseline_precise_b32/result.json")
    mid = read("midpoint_precise_b32/result.json")
    eul = read("euler_precise_b32/result.json")
    midpoint_max = read("maximum_batch_precise.json")["largest_stress_pass_batch"]
    euler_max = read("euler_maximum_batch_precise.json")["largest_stress_pass_batch"]
    baseline_max = read("baseline_capacity.json")["largest_stress_pass_batch"]
    memory = {
        "baseline_b32_allocated_gib": base["peak_allocated_bytes"] / GIB,
        "midpoint_b32_allocated_gib": mid["peak_allocated_bytes"] / GIB,
        "euler_b32_allocated_gib": eul["peak_allocated_bytes"] / GIB,
        "midpoint_saved_gib": (base["peak_allocated_bytes"] - mid["peak_allocated_bytes"]) / GIB,
        "euler_saved_gib": (base["peak_allocated_bytes"] - eul["peak_allocated_bytes"]) / GIB,
        "midpoint_saved_percent": 100 * (1 - mid["peak_allocated_bytes"] / base["peak_allocated_bytes"]),
        "euler_saved_percent": 100 * (1 - eul["peak_allocated_bytes"] / base["peak_allocated_bytes"]),
        "baseline_max_batch": baseline_max,
        "midpoint_max_batch": midpoint_max,
        "euler_max_batch": euler_max,
        "midpoint_capacity_multiple": midpoint_max / baseline_max,
        "euler_capacity_multiple": euler_max / baseline_max,
    }

    run_names = list(dict.fromkeys(
        [name for _, name in ORIGINAL] + [name for _, name in tuned] + list(SEED2.values())
    ))
    run_hours = {name: read(f"{name}/result.json")["training_seconds"] / 3600 for name in run_names}
    total_hours = sum(run_hours.values())
    run_wall_hours = {name: read(f"{name}/result.json")["wall_seconds"] / 3600 for name in run_names}
    total_wall_hours = sum(run_wall_hours.values())
    pilot_hours = sum(read(path.relative_to(ROOT))["training_seconds"] / 3600
                      for path in ROOT.glob("lrpilot_*/result.json"))
    pilot_wall_hours = sum(read(path.relative_to(ROOT))["wall_seconds"] / 3600
                           for path in ROOT.glob("lrpilot_*/result.json"))
    total_hours_with_pilots = total_hours + pilot_hours
    total_wall_hours_with_pilots = total_wall_hours + pilot_wall_hours
    gpu_watts_measured = 114.21
    laptop_watts_assumed = 160.0
    electricity_inr_per_kwh_assumed = 8.0
    grid_kg_co2e_per_kwh_assumed = 0.7
    model_parameters = 20_166_912
    supervised_tokens_per_run = 50_000_000
    cost = {
        "scope": "Original five, one additional tuned maximum-batch run, and three second-seed fixed-batch runs. Euler tuning retained its original 6e-4 run.",
        "measured_training_hours": total_hours,
        "measured_wall_hours": total_wall_hours,
        "lr_pilot_training_hours": pilot_hours,
        "lr_pilot_wall_hours": pilot_wall_hours,
        "measured_training_hours_including_lr_pilots": total_hours_with_pilots,
        "measured_wall_hours_including_lr_pilots": total_wall_hours_with_pilots,
        "run_training_hours": run_hours,
        "run_wall_hours": run_wall_hours,
        "measured_gpu_power_watts_during_midpoint_b100": gpu_watts_measured,
        "estimated_gpu_energy_kwh": total_wall_hours * gpu_watts_measured / 1000,
        "estimated_gpu_energy_kwh_including_lr_pilots": total_wall_hours_with_pilots * gpu_watts_measured / 1000,
        "assumed_whole_laptop_power_watts": laptop_watts_assumed,
        "whole_laptop_power_sensitivity_watts": [140, 200],
        "estimated_whole_laptop_energy_kwh": total_wall_hours * laptop_watts_assumed / 1000,
        "estimated_whole_laptop_energy_kwh_including_lr_pilots": total_wall_hours_with_pilots * laptop_watts_assumed / 1000,
        "assumed_electricity_inr_per_kwh": electricity_inr_per_kwh_assumed,
        "estimated_local_electricity_inr": total_wall_hours * laptop_watts_assumed / 1000 * electricity_inr_per_kwh_assumed,
        "estimated_local_electricity_inr_including_lr_pilots": total_wall_hours_with_pilots * laptop_watts_assumed / 1000 * electricity_inr_per_kwh_assumed,
        "local_cost_formula": "wall_hours * assumed_whole_laptop_kW * local_tariff_per_kWh",
        "assumed_grid_kg_co2e_per_kwh": grid_kg_co2e_per_kwh_assumed,
        "illustrative_operational_kg_co2e": total_wall_hours * laptop_watts_assumed / 1000 * grid_kg_co2e_per_kwh_assumed,
        "gcp_t4_gpu_only_usd_per_hour": 0.35,
        "illustrative_gcp_t4_gpu_only_usd": total_wall_hours * 0.35,
        "illustrative_gcp_t4_gpu_only_usd_including_lr_pilots": total_wall_hours_with_pilots * 0.35,
        "cloud_caveat": "GPU-only lower bound using published Google Cloud T4 on-demand price; excludes VM CPU/RAM/storage and does not adjust for T4 performance.",
        "colab_caveat": "Free Colab can have zero direct charge, but GPU type, limits, availability, and runtime duration are not guaranteed.",
        "thermal_caveat": "Second-seed and tuned-midpoint throughput was measured while the laptop was in Quiet Mode and thermally constrained. Those runs remain valid for loss, but their speed is excluded from controlled performance comparisons.",
        "approximate_6NT_flops_per_run": 6 * model_parameters * supervised_tokens_per_run,
        "approximate_6NT_pflops_per_run": 6 * model_parameters * supervised_tokens_per_run / 1e15,
        "flops_caveat": "6*N*T is a coarse dense-transformer training estimate; reversible recomputation, attention, padding and implementation details are not captured.",
    }

    report = {"seed_statistics": seed_stats, "memory_analysis": memory, "cost_estimate": cost,
              "data_compute_context": {
                  "parameters": model_parameters,
                  "training_tokens": supervised_tokens_per_run,
                  "tokens_per_parameter": supervised_tokens_per_run / model_parameters,
                  "illustrative_20_tokens_per_parameter_target": 20 * model_parameters,
                  "additional_tokens_to_20_per_parameter": 20 * model_parameters - supervised_tokens_per_run,
                  "note": "The 20 tokens/parameter value is a rough Chinchilla-era compute-optimal reference, not a quality guarantee or a fitted law for this tokenizer/model/data mixture."
              },
              "lr_tuning": tuning, "training_curve": "training_curves.png",
              "seed_errorbar_figure": "seed_loss_errorbars.png"}
    (ROOT / "extended_analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
