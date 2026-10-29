# Robust aggregation for federated intrusion detection (NSL-KDD)

Code and per-run results for the paper *"Robust Aggregation for Federated Intrusion Detection under
Heterogeneous Data and Model Poisoning: A Controlled Comparison on NSL-KDD"* (Pittan and Rajesh, VIT Vellore).

A pure-NumPy simulation of synchronous federated learning: ten clients train a 122-64-1 MLP on Dirichlet
partitions of NSL-KDD (α = 100 near-IID, α = 0.3 non-IID). 0%, 20% or 30% of clients send sign-flipped
updates scaled by γ = 3, and the server aggregates with FedAvg, equal-weight FedAvg, coordinate-wise median,
trimmed mean (β = 0.3) or Krum (f = 3). The misconfigured-defence ablation uses trimmed mean with β = 0.1 and
Krum with f = 1 against 30% attackers.

## Reproduce

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/defcom17/NSL_KDD.git /tmp/nsl
mkdir -p data && cp /tmp/nsl/KDDTrain+.txt /tmp/nsl/KDDTest+.txt data/
python src/run_all.py --data data --out results --workers 2     # 170 runs, about 15 min on 2 CPU cores
python src/analyze.py --results results --figures figures       # tables, statistics, figures 2-4
```

`run_all.py` skips runs whose log already exists, so delete `results/runs/` to rerun from scratch.
Each seed fixes the partition, the initial weights, the malicious clients and the mini-batch order,
so all rules see identical conditions for a given seed. Aggregation times depend on the machine.

## Layout

| Path | Contents |
|---|---|
| `src/fl_ids.py` | data loading, Dirichlet partition, MLP, local SGD, aggregation rules, one federated run |
| `src/run_all.py` | the 170 configurations in the paper |
| `src/analyze.py` | Tables 2–6, Welch t-tests with Holm correction, Cohen's d, Figures 2–4 |
| `results/runs/*.csv` | per-round accuracy, precision, recall, F1 and aggregation time of every run |
| `results/runs/*.json` | client sizes, attack share per client and malicious client ids of every run |
| `results/per_run_summary.csv` | final, mean-round and best-round metrics per run |
| `results/table_*.csv`, `results/summary.json` | numbers used in the paper |
| `figures/` | figures as used in the paper |
| `paper/` | scripts that build the manuscript from the result files |

Note: best-round accuracy is selected on the test set and is reported only to show how far final-round
accuracy falls below the peak. No validation split or early stopping is used.

## Citation

Pittan, A.R., Rajesh, R. Robust aggregation for federated intrusion detection under heterogeneous data and
model poisoning: a controlled comparison on NSL-KDD. (Under review.)

## Licence

Code: MIT (see `LICENSE`). NSL-KDD is distributed by the Canadian Institute for Cybersecurity, University of
New Brunswick, and is not included in this repository.
