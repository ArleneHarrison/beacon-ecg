"""Create the corrected beyond-echo mortality figure from observed follow-up."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def prepare_survival_groups(
    predictions: pd.DataFrame, quantile: float = 0.8
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    required = [
        "lvef_value",
        "pred_hfref_le40",
        "corrected_dead_365d",
        "outcome_observed_365d",
        "days_to_death",
    ]
    missing = [column for column in required if column not in predictions]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    observed = predictions["outcome_observed_365d"].fillna(False).astype(bool)
    frame = predictions.loc[
        observed
        & predictions["lvef_value"].gt(50)
        & predictions["pred_hfref_le40"].notna()
    ].copy()
    if frame["corrected_dead_365d"].isna().any():
        raise ValueError("observed rows contain missing corrected mortality labels")
    threshold = float(frame["pred_hfref_le40"].quantile(quantile))
    high = frame["pred_hfref_le40"].ge(threshold)
    frame["group"] = np.where(high, "top quintile", "remainder")
    frame["event"] = frame["corrected_dead_365d"].astype(int)
    death_time = pd.to_numeric(frame["days_to_death"], errors="coerce")
    if (frame["event"].eq(1) & death_time.isna()).any():
        raise ValueError("corrected death event lacks days_to_death")
    if (frame["event"].eq(1) & ~death_time.between(0, 365, inclusive="both")).any():
        raise ValueError("corrected death event lies outside the 365-day horizon")
    frame["time"] = np.where(frame["event"].eq(1), death_time, 365.0)
    high_frame = frame.loc[high]
    low_frame = frame.loc[~high]
    summary: dict[str, float | int] = {
        "n": int(len(frame)),
        "events": int(frame["event"].sum()),
        "threshold": threshold,
        "high_n": int(len(high_frame)),
        "low_n": int(len(low_frame)),
        "high_events": int(high_frame["event"].sum()),
        "low_events": int(low_frame["event"].sum()),
        "high_event_rate": float(high_frame["event"].mean()),
        "low_event_rate": float(low_frame["event"].mean()),
    }
    return frame.reset_index(drop=True), summary


def _p_label(value: float) -> str:
    return "<0.001" if value < 0.001 else f"={value:.3f}"


def make_figure(
    frame: pd.DataFrame,
    group_summary: dict[str, float | int],
    survival_result: dict[str, object],
    output: Path,
) -> dict[str, object]:
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    high = frame.loc[frame["group"].eq("top quintile")]
    low = frame.loc[frame["group"].eq("remainder")]
    logrank = logrank_test(
        high["time"], low["time"], event_observed_A=high["event"], event_observed_B=low["event"]
    )
    full = survival_result["median_imputed_full_year"]
    early = survival_result["piecewise_models"]["days_0_to_90"]
    late = survival_result["piecewise_models"]["days_91_to_365"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
        }
    )
    orange, green = "#D55E00", "#009E73"
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15), constrained_layout=True)
    ax = axes[0]
    rates = [100 * float(group_summary["high_event_rate"]), 100 * float(group_summary["low_event_rate"])]
    bars = ax.bar(
        [f"Top quintile\n(n={int(group_summary['high_n']):,})", f"Remainder\n(n={int(group_summary['low_n']):,})"],
        rates,
        color=[orange, green],
        edgecolor="black",
        linewidth=0.5,
        width=0.58,
    )
    for bar, value in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("One-year all-cause mortality (%)")
    ax.set_ylim(0, max(rates) * 1.25)
    ax.set_title("a  Observed event rates", loc="left", fontweight="bold")
    ax.text(0.5, 0.98, f"LVEF >50%, n={int(group_summary['n']):,}", transform=ax.transAxes, ha="center", va="top", fontsize=7, style="italic")

    ax = axes[1]
    for subset, colour, label in [
        (high, orange, "Top structural-risk quintile"),
        (low, green, "Remaining four quintiles"),
    ]:
        km = KaplanMeierFitter()
        km.fit(subset["time"], event_observed=subset["event"], label=label)
        km.plot_survival_function(ax=ax, ci_show=True, color=colour, linewidth=1.4, show_censors=False)
    ax.set_xlim(0, 365)
    ax.set_ylim(0.70, 1.00)
    ax.set_xlabel("Days from index ECG")
    ax.set_ylabel("Survival probability")
    ax.set_title("b  Kaplan–Meier analysis", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    annotation = (
        f"Log-rank P {_p_label(float(logrank.p_value))}\n"
        f"Adjusted average HR {full['high_risk_hr']:.2f} "
        f"({full['high_risk_ci_lower']:.2f}–{full['high_risk_ci_upper']:.2f})\n"
        f"0–90 d HR {early['high_risk_hr']:.2f}; 91–365 d HR {late['high_risk_hr']:.2f}"
    )
    ax.text(0.98, 0.98, annotation, transform=ax.transAxes, ha="right", va="top", fontsize=6.8)
    ax.text(
        0.98,
        0.02,
        "Y-axis truncated at 0.70",
        transform=ax.transAxes,
        ha="right",
        fontsize=6,
        color="#555555",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {**group_summary, "logrank_p": float(logrank.p_value)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--survival-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions)
    survival = json.loads(args.survival_json.read_text(encoding="utf-8"))
    frame, group_summary = prepare_survival_groups(predictions)
    summary = make_figure(frame, group_summary, survival, args.output)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
