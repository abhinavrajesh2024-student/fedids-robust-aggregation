"""Federated intrusion detection on NSL-KDD with robust aggregation.

Pure-NumPy implementation of the experiment described in the paper:
a 122-64-1 MLP trained by K = 10 simulated clients, aggregated with
FedAvg, equal-weight FedAvg, coordinate-wise median, trimmed mean or Krum,
with a fraction of clients sending scaled sign-flipped updates.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]

DOS = {"back", "land", "neptune", "pod", "smurf", "teardrop"}
PROBE = {"ipsweep", "nmap", "portsweep", "satan"}
R2L = {"ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster"}
U2R = {"buffer_overflow", "loadmodule", "perl", "rootkit"}
CATEGORY_NAMES = ["normal", "dos", "probe", "r2l", "u2r"]


def _category(label: str) -> int:
    if label == "normal":
        return 0
    if label in DOS:
        return 1
    if label in PROBE:
        return 2
    if label in R2L:
        return 3
    if label in U2R:
        return 4
    raise ValueError(f"unknown training label {label}")


@dataclass
class Data:
    Xtr: np.ndarray
    ytr: np.ndarray
    ctr: np.ndarray  # five-way category, used only for partitioning
    Xte: np.ndarray
    yte: np.ndarray


def load_nsl_kdd(data_dir: str | Path) -> Data:
    data_dir = Path(data_dir)
    tr = pd.read_csv(data_dir / "KDDTrain+.txt", header=None, names=COLS)
    te = pd.read_csv(data_dir / "KDDTest+.txt", header=None, names=COLS)
    both = pd.concat([tr, te], keys=["tr", "te"])
    feats = pd.get_dummies(both.drop(columns=["label", "difficulty"]), columns=CATEGORICAL, dtype=float)
    Xtr_raw = feats.loc["tr"].to_numpy(dtype=np.float64)
    Xte_raw = feats.loc["te"].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(Xtr_raw)  # fitted on the training set only
    Xtr = scaler.transform(Xtr_raw).astype(np.float32)
    Xte = scaler.transform(Xte_raw).astype(np.float32)
    ytr = (tr["label"] != "normal").to_numpy(dtype=np.float32)
    yte = (te["label"] != "normal").to_numpy(dtype=np.float32)
    ctr = tr["label"].map(_category).to_numpy()
    return Data(Xtr, ytr, ctr, Xte, yte)


def dirichlet_partition(categories: np.ndarray, K: int, alpha: float, rng: np.random.Generator,
                        min_size: int = 128) -> list[np.ndarray]:
    """Per-category Dirichlet split (Hsu et al., 2019). Redraws until every client
    holds at least ``min_size`` records (one mini-batch)."""
    while True:
        parts: list[list[np.ndarray]] = [[] for _ in range(K)]
        for c in np.unique(categories):
            idx = np.flatnonzero(categories == c)
            rng.shuffle(idx)
            p = rng.dirichlet(np.full(K, alpha))
            cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].append(chunk)
        shards = [np.sort(np.concatenate(p)) for p in parts]
        if min(len(s) for s in shards) >= min_size:
            return shards


# ----------------------------------------------------------------------------- model
D_IN, D_H = 122, 64
N_PARAMS = D_IN * D_H + D_H + D_H + 1  # 7,937


def init_params(rng: np.random.Generator) -> np.ndarray:
    W1 = rng.normal(0.0, np.sqrt(2.0 / D_IN), size=(D_IN, D_H))
    W2 = rng.normal(0.0, np.sqrt(2.0 / D_H), size=(D_H, 1))
    return np.concatenate([W1.ravel(), np.zeros(D_H), W2.ravel(), np.zeros(1)]).astype(np.float32)


def unpack(w: np.ndarray):
    i = 0
    W1 = w[i:i + D_IN * D_H].reshape(D_IN, D_H); i += D_IN * D_H
    b1 = w[i:i + D_H]; i += D_H
    W2 = w[i:i + D_H].reshape(D_H, 1); i += D_H
    b2 = w[i:i + 1]
    return W1, b1, W2, b2


def predict_proba(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    W1, b1, W2, b2 = unpack(w)
    h = np.maximum(X @ W1 + b1, 0.0)
    z = (h @ W2 + b2).ravel()
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def local_train(w: np.ndarray, X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                lr: float = 0.05, batch: int = 128, epochs: int = 1) -> np.ndarray:
    w = w.copy()
    W1, b1, W2, b2 = unpack(w)  # views into w, updated in place
    n = len(y)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            b = order[s:s + batch]
            xb, yb = X[b], y[b]
            a1 = xb @ W1 + b1
            h = np.maximum(a1, 0.0)
            z = (h @ W2 + b2).ravel()
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
            dz = ((p - yb) / len(b))[:, None]           # dL/dz for mean BCE
            gW2 = h.T @ dz
            gb2 = dz.sum(0)
            dh = (dz @ W2.T) * (a1 > 0)
            gW1 = xb.T @ dh
            gb1 = dh.sum(0)
            W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    return w


# ----------------------------------------------------------------------------- aggregation
def agg_fedavg(U: np.ndarray, n: np.ndarray, **_) -> np.ndarray:
    return (n[:, None] * U).sum(0) / n.sum()


def agg_fedavg_equal(U: np.ndarray, **_) -> np.ndarray:
    return U.mean(0)


def agg_median(U: np.ndarray, **_) -> np.ndarray:
    return np.median(U, axis=0)


def agg_trimmed_mean(U: np.ndarray, beta: float = 0.3, **_) -> np.ndarray:
    K = U.shape[0]
    m = int(np.floor(beta * K))
    S = np.sort(U, axis=0)
    return S[m:K - m].mean(0)


def agg_krum(U: np.ndarray, f: int = 3, **_) -> np.ndarray:
    K = U.shape[0]
    sq = (U * U).sum(1)
    D = sq[:, None] + sq[None, :] - 2.0 * (U @ U.T)
    np.fill_diagonal(D, np.inf)
    nn = K - f - 2
    scores = np.sort(D, axis=1)[:, :nn].sum(1)
    return U[int(np.argmin(scores))]


RULES = {
    "fedavg": agg_fedavg,
    "fedavg_equal": agg_fedavg_equal,
    "median": agg_median,
    "trimmed_mean": agg_trimmed_mean,
    "krum": agg_krum,
}


def evaluate(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> dict:
    pred = (predict_proba(w, X) >= 0.5).astype(np.float32)
    return {
        "accuracy": 100 * accuracy_score(y, pred),
        "precision": 100 * precision_score(y, pred, zero_division=0),
        "recall": 100 * recall_score(y, pred, zero_division=0),
        "f1": 100 * f1_score(y, pred, zero_division=0),
    }


def run(data: Data, *, rule: str, alpha: float, malicious_frac: float, seed: int,
        K: int = 10, rounds: int = 25, gamma: float = 3.0, beta: float = 0.3, f: int = 3,
        lr: float = 0.05, batch: int = 128) -> tuple[pd.DataFrame, dict]:
    """One federated run. The seed fixes the partition, the initial weights, the
    malicious clients and the mini-batch order; these are identical across rules."""
    shards = dirichlet_partition(data.ctr, K, alpha, np.random.default_rng([seed, 0]))
    w = init_params(np.random.default_rng([seed, 1]))
    n_mal = int(round(malicious_frac * K))
    malicious = set(np.random.default_rng([seed, 2]).choice(K, size=n_mal, replace=False).tolist())
    n = np.array([len(s) for s in shards], dtype=np.float64)
    agg = RULES[rule]
    rows = []
    for t in range(1, rounds + 1):
        U = np.empty((K, N_PARAMS), dtype=np.float32)
        for k, idx in enumerate(shards):
            wk = local_train(w, data.Xtr[idx], data.ytr[idx], np.random.default_rng([seed, 3, t, k]),
                             lr=lr, batch=batch)
            d = wk - w
            U[k] = -gamma * d if k in malicious else d
        t0 = time.perf_counter()
        step = agg(U, n=n, beta=beta, f=f)
        agg_ms = 1000 * (time.perf_counter() - t0)
        w = (w + step).astype(np.float32)
        rows.append({"round": t, **evaluate(w, data.Xte, data.yte), "agg_ms": agg_ms})
    meta = {
        "client_sizes": n.astype(int).tolist(),
        "client_attack_share": [float(data.ytr[s].mean()) for s in shards],
        "malicious_clients": sorted(malicious),
    }
    return pd.DataFrame(rows), meta
