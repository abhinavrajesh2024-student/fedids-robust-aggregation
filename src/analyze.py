"""Build every table, statistic and figure in the paper from results/runs/.

Usage: python src/analyze.py --results results --figures figures
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

RULE_LABEL = {"fedavg": "FedAvg", "fedavg_equal": "FedAvg (equal weights)", "median": "Median",
              "trimmed_mean": "Trimmed mean", "krum": "Krum"}
MAIN_RULES = ["fedavg", "median", "trimmed_mean", "krum"]
PART_LABEL = {"near_iid": "Near-IID", "non_iid": "Non-IID"}


def load(results: Path) -> pd.DataFrame:
    frames = [pd.read_csv(f) for f in sorted(glob.glob(str(results / "runs" / "*.csv")))]
    df = pd.concat(frames, ignore_index=True)
    df["mal"] = (df["malicious_frac"] * 100).round().astype(int)
    return df


def per_run(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["experiment", "partition", "mal", "rule", "beta", "f", "seed"]
    g = df.groupby(keys)
    final = g.apply(lambda x: x.sort_values("round").iloc[-1], include_groups=False)
    out = final[["accuracy", "f1", "recall", "precision"]].copy()
    out["mean_round_acc"] = g["accuracy"].mean()
    out["best_acc"] = g["accuracy"].max()
    out["best_round"] = g.apply(lambda x: int(x.loc[x["accuracy"].idxmax(), "round"]), include_groups=False)
    out["round_to_74"] = g.apply(
        lambda x: (x.loc[x["accuracy"] >= 74, "round"].min() if (x["accuracy"] >= 74).any() else np.nan),
        include_groups=False)
    out["agg_ms"] = g["agg_ms"].mean()
    return out.reset_index()


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    a = ap.parse_args()
    R, FIG = Path(a.results), Path(a.figures)
    FIG.mkdir(exist_ok=True, parents=True)
    df = load(R)
    runs = per_run(df)
    runs.to_csv(R / "per_run_summary.csv", index=False)
    main_runs = runs[runs.experiment == "main"]
    S = {}

    # ---- Tables 2 and 3: final-round performance
    rows = []
    for part in PART_LABEL:
        for mal in (0, 20, 30):
            for rule in MAIN_RULES + ["fedavg_equal"]:
                x = main_runs[(main_runs.partition == part) & (main_runs.mal == mal) & (main_runs.rule == rule)]
                rows.append(dict(partition=part, mal=mal, rule=rule, n=len(x),
                                 acc_mean=x.accuracy.mean(), acc_sd=x.accuracy.std(),
                                 f1_mean=x.f1.mean(), f1_sd=x.f1.std(),
                                 recall=x.recall.mean(), precision=x.precision.mean(),
                                 acc_min=x.accuracy.min(), acc_max=x.accuracy.max(),
                                 mean_round_acc=x.mean_round_acc.mean(),
                                 best_acc_mean=x.best_acc.mean(), best_acc_sd=x.best_acc.std(),
                                 best_round_mean=x.best_round.mean(),
                                 r74_mean=x.round_to_74.mean(), r74_n=int(x.round_to_74.notna().sum()),
                                 agg_ms=x.agg_ms.mean()))
    perf = pd.DataFrame(rows)
    perf.to_csv(R / "table_performance.csv", index=False)

    # ---- Table 4: robust rule vs FedAvg, Welch t, Holm, Cohen's d
    comp = []
    for part in PART_LABEL:
        for mal in (20, 30):
            base = main_runs[(main_runs.partition == part) & (main_runs.mal == mal) & (main_runs.rule == "fedavg")]
            for rule in ["median", "trimmed_mean", "krum"]:
                x = main_runs[(main_runs.partition == part) & (main_runs.mal == mal) & (main_runs.rule == rule)]
                t = stats.ttest_ind(x.accuracy, base.accuracy, equal_var=False)
                comp.append(dict(partition=part, mal=mal, rule=rule,
                                 delta=x.accuracy.mean() - base.accuracy.mean(),
                                 p=t.pvalue, d=cohens_d(x.accuracy, base.accuracy)))
    comp = pd.DataFrame(comp)
    comp["p_holm"] = holm(comp.p)
    comp.to_csv(R / "table_stats.csv", index=False)

    # ---- equal-weight FedAvg vs weighted FedAvg and vs robust rules, non-IID clean
    eq = {}
    for part in PART_LABEL:
        for mal in (0, 20, 30):
            sel = lambda r: main_runs[(main_runs.partition == part) & (main_runs.mal == mal) & (main_runs.rule == r)].sort_values("seed")
            e, w = sel("fedavg_equal"), sel("fedavg")
            t = stats.ttest_ind(e.accuracy, w.accuracy, equal_var=False)
            eq[f"{part}_{mal}"] = dict(eq_acc=e.accuracy.mean(), eq_sd=e.accuracy.std(), w_acc=w.accuracy.mean(),
                                       delta=e.accuracy.mean() - w.accuracy.mean(), p=t.pvalue,
                                       eq_recall=e.recall.mean(), eq_per_seed=e.accuracy.round(1).tolist())
    S["equal_weight"] = eq

    # ---- misconfigured defence
    mis = []
    for part in PART_LABEL:
        for rule, beta, f in (("trimmed_mean", 0.1, 3), ("krum", 0.3, 1)):
            x = runs[(runs.experiment == "misconfigured") & (runs.partition == part) & (runs.rule == rule)]
            ok = main_runs[(main_runs.partition == part) & (main_runs.mal == 30) & (main_runs.rule == rule)]
            fa = main_runs[(main_runs.partition == part) & (main_runs.mal == 30) & (main_runs.rule == "fedavg")]
            mis.append(dict(partition=part, rule=rule, setting=("β = 0.1 (m = 1)" if rule == "trimmed_mean" else "f = 1"),
                            acc_mean=x.accuracy.mean(), acc_sd=x.accuracy.std(), f1_mean=x.f1.mean(), f1_sd=x.f1.std(),
                            recall=x.recall.mean(), precision=x.precision.mean(),
                            acc_min=x.accuracy.min(), acc_max=x.accuracy.max(),
                            correct_acc=ok.accuracy.mean(), correct_sd=ok.accuracy.std(),
                            fedavg_acc=fa.accuracy.mean(),
                            per_seed=x.sort_values("seed").accuracy.round(1).tolist()))
    pd.DataFrame(mis).to_csv(R / "table_misconfigured.csv", index=False)

    # ---- aggregation time over all main configurations
    tm = main_runs.groupby("rule").agg_ms.agg(["mean", "std"])
    S["agg_ms"] = {r: dict(mean=float(tm.loc[r, "mean"]), sd=float(tm.loc[r, "std"])) for r in tm.index}
    S["agg_ratio"] = {r: float(tm.loc[r, "mean"] / tm.loc["fedavg", "mean"]) for r in tm.index}

    # ---- learning dynamics: round-1 vs round-25 on non-IID clean
    dyn = {}
    for part in PART_LABEL:
        for rule in MAIN_RULES + ["fedavg_equal"]:
            x = df[(df.experiment == "main") & (df.partition == part) & (df.mal == 0) & (df.rule == rule)]
            m = x.groupby("round").accuracy.mean()
            dyn[f"{part}_{rule}"] = dict(r1=float(m.loc[1]), r25=float(m.loc[25]), peak=float(m.max()),
                                         peak_round=int(m.idxmax()))
    S["dynamics"] = dyn

    # ---- per-seed notes used in the text
    notes = {}
    for part in PART_LABEL:
        for mal in (0, 20, 30):
            for rule in MAIN_RULES + ["fedavg_equal"]:
                x = main_runs[(main_runs.partition == part) & (main_runs.mal == mal) & (main_runs.rule == rule)].sort_values("seed")
                notes[f"{part}_{mal}_{rule}"] = x.accuracy.round(2).tolist()
    S["per_seed_final_acc"] = notes
    meta = json.loads((R / "runs" / "main_non_iid_mal0_fedavg_b0.3_f3_seed0.json").read_text())
    S["seed0_non_iid_sizes"] = meta["client_sizes"]
    S["seed0_non_iid_largest_share"] = max(meta["client_sizes"]) / sum(meta["client_sizes"])
    (R / "summary.json").write_text(json.dumps(S, indent=2))

    make_figures(df, perf, main_runs, FIG)
    print(perf.round(2).to_string())
    print(comp.round(4).to_string())
    print(pd.DataFrame(mis).round(2).to_string())
    print(json.dumps({k: S[k] for k in ("equal_weight", "agg_ms", "agg_ratio", "dynamics")}, indent=1))


STYLE = {"fedavg": dict(color="#4C72B0", marker="o", ls="-", hatch=""),
         "median": dict(color="#DD8452", marker="s", ls="--", hatch="//"),
         "trimmed_mean": dict(color="#55A868", marker="^", ls="-.", hatch="\\\\"),
         "krum": dict(color="#C44E52", marker="D", ls=":", hatch="xx"),
         "fedavg_equal": dict(color="#8172B3", marker="v", ls=(0, (5, 1)), hatch="..")}


def make_figures(df, perf, main_runs, FIG):
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    # Figure 2: final accuracy bars
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, part in zip(axes, PART_LABEL):
        w = 0.2
        for i, rule in enumerate(MAIN_RULES):
            p = perf[(perf.partition == part) & (perf.rule == rule)].sort_values("mal")
            ax.bar(np.arange(3) + (i - 1.5) * w, p.acc_mean, w, yerr=p.acc_sd, capsize=3,
                   color=STYLE[rule]["color"], hatch=STYLE[rule]["hatch"], edgecolor="black", lw=0.5,
                   label=RULE_LABEL[rule])
        ax.set_xticks(range(3), ["0%", "20%", "30%"])
        ax.set_xlabel("Malicious clients")
        ax.set_title(f"{PART_LABEL[part]} ({'α = 100' if part == 'near_iid' else 'α = 0.3'})")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Final test accuracy (%)")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_final_accuracy.png", dpi=200)
    plt.close(fig)

    # Figure 3: non-IID curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, mal in zip(axes, (0, 20, 30)):
        for rule in MAIN_RULES:
            x = df[(df.experiment == "main") & (df.partition == "non_iid") & (df.mal == mal) & (df.rule == rule)]
            g = x.groupby("round").accuracy
            m, s = g.mean(), g.std()
            st = STYLE[rule]
            ax.plot(m.index, m, color=st["color"], ls=st["ls"], marker=st["marker"], ms=4, markevery=3,
                    label=RULE_LABEL[rule])
            ax.fill_between(m.index, m - s, m + s, color=st["color"], alpha=0.12)
        ax.set_title(f"Non-IID, {mal}% malicious")
        ax.set_xlabel("Communication round")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_noniid_curves.png", dpi=200)
    plt.close(fig)

    # Figure 4: near-IID 30% curves + aggregation time
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for rule in MAIN_RULES:
        x = df[(df.experiment == "main") & (df.partition == "near_iid") & (df.mal == 30) & (df.rule == rule)]
        g = x.groupby("round").accuracy
        m, s = g.mean(), g.std()
        st = STYLE[rule]
        ax.plot(m.index, m, color=st["color"], ls=st["ls"], marker=st["marker"], ms=4, markevery=3, label=RULE_LABEL[rule])
        ax.fill_between(m.index, m - s, m + s, color=st["color"], alpha=0.12)
    ax.set_title("(a) Near-IID, 30% malicious")
    ax.set_xlabel("Communication round"); ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(0, 100)
    ax.grid(alpha=0.3); ax.legend(loc="center right")
    ax = axes[1]
    t = main_runs[main_runs.rule.isin(MAIN_RULES)].groupby("rule").agg_ms.agg(["mean", "std"]).loc[MAIN_RULES]
    ax.bar(range(4), t["mean"], yerr=t["std"], capsize=4, color=[STYLE[r]["color"] for r in MAIN_RULES],
           hatch=None, edgecolor="black", lw=0.5)
    for i, r in enumerate(MAIN_RULES):
        ax.patches[i].set_hatch(STYLE[r]["hatch"])
    ax.set_xticks(range(4), [RULE_LABEL[r] for r in MAIN_RULES])
    ax.set_ylabel("Aggregation time per round (ms)")
    ax.set_title("(b) Server aggregation time")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_nearIID30_and_time.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
