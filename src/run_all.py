"""Run every configuration reported in the paper and write per-round logs.

Usage: python src/run_all.py --data data --out results [--workers 2]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fl_ids as F  # noqa: E402

PARTITIONS = {"near_iid": 100.0, "non_iid": 0.3}
SEEDS = range(5)


def configs():
    for part in PARTITIONS:
        for frac in (0.0, 0.2, 0.3):
            for rule in ("fedavg", "median", "trimmed_mean", "krum", "fedavg_equal"):
                for s in SEEDS:
                    yield dict(experiment="main", partition=part, malicious_frac=frac, rule=rule,
                               beta=0.3, f=3, seed=s)
        # misconfigured defence: defender under-estimates the attacker fraction
        for rule, beta, f in (("trimmed_mean", 0.1, 3), ("krum", 0.3, 1)):
            for s in SEEDS:
                yield dict(experiment="misconfigured", partition=part, malicious_frac=0.3, rule=rule,
                           beta=beta, f=f, seed=s)


def run_name(c):
    return (f"{c['experiment']}_{c['partition']}_mal{int(c['malicious_frac']*100)}_{c['rule']}"
            f"_b{c['beta']}_f{c['f']}_seed{c['seed']}")


_DATA = None


def _work(args):
    c, data_dir, out = args
    global _DATA
    if _DATA is None:
        _DATA = F.load_nsl_kdd(data_dir)
    path = Path(out) / "runs" / f"{run_name(c)}.csv"
    if path.exists():
        return run_name(c)
    df, meta = F.run(_DATA, rule=c["rule"], alpha=PARTITIONS[c["partition"]],
                     malicious_frac=c["malicious_frac"], seed=c["seed"], beta=c["beta"], f=c["f"])
    for k, v in c.items():
        df[k] = v
    df.to_csv(path, index=False)
    (Path(out) / "runs" / f"{run_name(c)}.json").write_text(json.dumps({**c, **meta}))
    return run_name(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    (Path(a.out) / "runs").mkdir(parents=True, exist_ok=True)
    jobs = [(c, a.data, a.out) for c in configs()]
    with ProcessPoolExecutor(a.workers) as ex:
        for i, name in enumerate(ex.map(_work, jobs), 1):
            print(f"[{i}/{len(jobs)}] {name}", flush=True)


if __name__ == "__main__":
    main()
