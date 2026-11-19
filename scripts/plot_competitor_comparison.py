"""Plot accuracy, loss and simulated systems metrics from competitor trial JSON."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "research"
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_NAME = "competitor_benchmark_results.json"
FIGURE_NAME = "fig_competitor_comparison.png"

STYLES = {
    "FedAvg": ("#555555", "o"),
    "FedProx": ("#0072B2", "s"),
    "FedNova": ("#D55E00", "^"),
    "CapDrift-Legacy": ("#8C5FA8", "v"),
    "CapDrift-FL": ("#008065", "D"),
}


def _load_results(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    # Accept both the historical flat JSON and a metadata + methods envelope.
    methods = data.get("methods", data)
    trials_by_method = {}
    rounds = None
    for method, info in methods.items():
        if not isinstance(info, dict) or not {"trials", "history"}.intersection(info):
            continue
        trials = info["trials"] if "trials" in info else [info]
        if not trials:
            raise ValueError(f"{method} has no trials")
        for trial in trials:
            trial_rounds = [record["round"] for record in trial["history"]]
            if not trial_rounds or trial_rounds != sorted(set(trial_rounds)):
                raise ValueError(f"{method} has missing, duplicate or unordered round records")
            if rounds is None:
                rounds = trial_rounds
            elif rounds != trial_rounds:
                raise ValueError(f"{method} does not share the benchmark round grid")
        trials_by_method[method] = trials
    if not trials_by_method:
        raise ValueError(f"No method trials found in {path}")
    return trials_by_method, rounds, data.get("metadata", {})


def _history_stats(trials, field):
    histories = [trial["history"] for trial in trials]
    if not all(field in record for history in histories for record in history):
        return None
    values = np.asarray([[record[field] for record in history] for history in histories])
    return values.mean(axis=0), values.std(axis=0)


def _trial_stats(trials, field):
    values = [trial[field] for trial in trials]
    return float(np.mean(values)), float(np.std(values))


def _seed_label(methods):
    seeds = [[trial.get("seed") for trial in trials] for trials in methods.values()]
    first = seeds[0]
    if all(None not in ids and len(set(ids)) == len(ids) and set(ids) == set(first) for ids in seeds):
        suffix = "s" if len(first) != 1 else ""
        return f"{len(first)} shared seed{suffix}: {', '.join(map(str, first))}"
    if all(all(seed is None for seed in ids) for ids in seeds):
        return "seed IDs not recorded"
    return "seed sets differ or are incomplete"


def plot_competitors(results_path=None, output_path=None, *, metadata_path=None):
    """Write the figure, retaining the historic docs copies on a default run."""
    uses_default_input = results_path is None
    copy_to_docs = uses_default_input and output_path is None
    results_path = Path(results_path) if results_path is not None else RESULTS_DIR / RESULTS_NAME
    default_plot_dir = PLOTS_DIR if uses_default_input else results_path.parent / "plots"
    output_path = Path(output_path) if output_path is not None else default_plot_dir / FIGURE_NAME
    methods, rounds, metadata = _load_results(results_path)
    if metadata_path is not None:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    elif results_path.name == RESULTS_NAME:
        sidecar = results_path.with_name("competitor_benchmark_metadata.json")
        if sidecar.exists():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))

    all_times = [trial["total_time_sec"] for trials in methods.values() for trial in trials]
    expected_time = metadata.get("modeled_reference_total_time_sec")
    if len(set(all_times)) == 1:
        time_summary = f"{all_times[0]:.2f} s in every trial"
    else:
        time_summary = f"{min(all_times):.2f}–{max(all_times):.2f} s across trials"

    with plt.rc_context({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 300}):
        fig, axes = plt.subplots(2, 3, figsize=(16, 8.8), layout="constrained")
        accuracy_ax, loss_ax, util_ax, pareto_ax, idle_ax, time_ax = axes.flat
        description = f"{len(rounds)} rounds · {_seed_label(methods)} · Modeled runtime: {time_summary}"
        alpha = metadata.get("alpha")
        if alpha is not None:
            description += f" · Dirichlet α={alpha}"
        fig.suptitle(f"Federated comparison under hardware and label skew\n{description}", fontsize=13)

        curves = (
            (accuracy_ax, "val_accuracy", "(a) Accuracy convergence", "Validation accuracy (%)"),
            (loss_ax, "val_loss", "(b) Cross-entropy convergence", "Validation cross-entropy"),
            (util_ax, "cluster_utilization_pct", "(c) Modeled cluster utilization", "Cluster utilization (%)"),
        )
        for ax, _, title, ylabel in curves:
            ax.set(title=title, xlabel="Communication round", ylabel=ylabel)
        accuracy_ax.set_ylim(0, 100)
        util_ax.set_ylim(0, 100)

        idle_max = 0.0
        drawn_metrics = set()
        elapsed_bounds = list(all_times)
        for position, (method, trials) in enumerate(methods.items()):
            color, marker = STYLES.get(method, (f"C{position % 10}", "o"))
            for ax, field, _, _ in curves:
                stats = _history_stats(trials, field)
                if stats is not None:
                    drawn_metrics.add(field)
                    mean, std = stats
                    ax.plot(rounds, mean, color=color, marker=marker, markersize=4,
                            markevery=max(1, len(rounds) // 6), label=method)
                    ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.13)

            accuracy, accuracy_sd = _trial_stats(trials, "final_accuracy")
            elapsed, elapsed_sd = _trial_stats(trials, "total_time_sec")
            elapsed_bounds.extend((elapsed - elapsed_sd, elapsed + elapsed_sd))
            idle, idle_sd = _trial_stats(trials, "total_idle_sec")
            pareto_ax.errorbar(elapsed, accuracy, xerr=elapsed_sd, yerr=accuracy_sd, color=color,
                               fmt=marker, capsize=3, markersize=6,
                               label=f"{method}  {accuracy:.1f}%")
            idle_ax.bar(position, idle, yerr=idle_sd, color=color, capsize=3, width=0.6)
            idle_ax.annotate(f"{idle:.1f}", (position, idle + idle_sd), xytext=(0, 4),
                             textcoords="offset points", ha="center")
            idle_max = max(idle_max, idle + idle_sd)
            time_ax.errorbar(position, elapsed, yerr=elapsed_sd, color=color, fmt=marker,
                             capsize=3, markersize=6)
            time_ax.annotate(f"{elapsed:.2f}", (position, elapsed + elapsed_sd), xytext=(0, 6),
                             textcoords="offset points", ha="center")

        for ax, field, _, _ in curves:
            if field not in drawn_metrics:
                ax.text(0.5, 0.5, "Metric not recorded", transform=ax.transAxes, ha="center")
        loss_ax.set_ylim(bottom=0)
        if "val_accuracy" in drawn_metrics:
            accuracy_ax.legend(loc="lower right", fontsize=8, title="Mean ± seed SD")
        pareto_ax.set(title="(d) Final accuracy vs. modeled runtime", xlabel="Modeled runtime (s)",
                      ylabel="Final validation accuracy (%)", ylim=(0, 100))
        pareto_ax.legend(loc="lower right", fontsize=8)
        pareto_ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        idle_ax.set(title="(e) Synchronization idle waste", ylabel="Aggregate client idle (s)",
                    ylim=(0, max(1, idle_max * 1.16)))
        time_ax.set(title="(f) Modeled completion time", ylabel="Modeled runtime (s)")
        for ax in (idle_ax, time_ax):
            ax.set_xticks(range(len(methods)), list(methods), rotation=16, ha="right")
            ax.set_xlim(-0.55, len(methods) - 0.45)
        if expected_time is not None:
            time_ax.axhline(expected_time, color="#333333", linestyle=":", zorder=0,
                            label=f"Modeled reference: {expected_time:.2f} s")
            time_ax.legend(loc="lower right", fontsize=8)
        # Include mean ± SD, observed range and reference without misleading x-axis jitter.
        if expected_time is not None:
            elapsed_bounds.append(expected_time)
        lo, hi = min(elapsed_bounds), max(elapsed_bounds)
        margin = max(0.2, (hi - lo) * 0.2)
        time_ax.set_ylim(lo - margin, hi + margin)
        time_ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        pareto_ax.set_xlim(lo - margin, hi + margin)
        for ax in axes.flat:
            ax.set_axisbelow(True)
            ax.grid(axis="y", linestyle="--", alpha=0.35)

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path)
        finally:
            plt.close(fig)

    if copy_to_docs:
        for destination in (
            PROJECT_ROOT / "docs" / "figures" / FIGURE_NAME,
            PROJECT_ROOT / "docs" / FIGURE_NAME,
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output_path, destination)
    print(f"[Generated Competitor Benchmark Figure] -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Competitor benchmark JSON")
    parser.add_argument("--output", type=Path, help="Figure output path (omit for the standard location)")
    parser.add_argument("--metadata", type=Path, help="Optional benchmark metadata JSON")
    args = parser.parse_args()
    try:
        plot_competitors(args.input, args.output, metadata_path=args.metadata)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
