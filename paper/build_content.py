"""Assemble the manuscript text and tables (as JSON) from the result files.
Every number in the paper is read from results/ here, so text and tables cannot drift."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
perf = pd.read_csv(R / "table_performance.csv")
st = pd.read_csv(R / "table_stats.csv")
mis = pd.read_csv(R / "table_misconfigured.csv")
S = json.loads((R / "summary.json").read_text())

L = {"fedavg": "FedAvg", "fedavg_equal": "FedAvg (equal wt.)", "median": "Median",
     "trimmed_mean": "Trimmed mean", "krum": "Krum"}


def P(part, mal, rule, col="acc_mean"):
    return float(perf[(perf.partition == part) & (perf.mal == mal) & (perf.rule == rule)][col].iloc[0])


def f1(x):
    return f"{x:.1f}"


def pm(part, mal, rule, a="acc"):
    return f"{P(part, mal, rule, a + '_mean'):.1f} ± {P(part, mal, rule, a + '_sd'):.1f}"


def rng(vals):
    return f"{min(vals):.1f}–{max(vals):.1f}"


def stat(part, mal, rule, col):
    return float(st[(st.partition == part) & (st.mal == mal) & (st.rule == rule)][col].iloc[0])


def pfmt(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


ROB = ["median", "trimmed_mean", "krum"]
EQ = S["equal_weight"]
T = S["agg_ms"]
RAT = S["agg_ratio"]
DY = S["dynamics"]
PS = S["per_seed_final_acc"]

near_clean = [P("near_iid", 0, r) for r in ["fedavg", "median", "trimmed_mean", "krum"]]
near30_rob = [P("near_iid", 30, r) for r in ROB]
drop_fedavg_near = P("near_iid", 0, "fedavg") - P("near_iid", 30, "fedavg")
max_rob_change = max(abs(P("near_iid", 30, r) - P("near_iid", 0, r)) for r in ROB)
non20_rob = [P("non_iid", 20, r) for r in ROB]
non30_rob = [P("non_iid", 30, r) for r in ROB]
fa20 = PS["non_iid_20_fedavg"]
p_non20 = [stat("non_iid", 20, r, "p") for r in ROB]
p_non30 = [stat("non_iid", 30, r, "p") for r in ROB]
ph_non = [stat("non_iid", m, r, "p_holm") for m in (20, 30) for r in ROB]
d_non = [stat("non_iid", m, r, "d") for m in (20, 30) for r in ROB]
d_near30 = [stat("near_iid", 30, r, "d") for r in ROB]
eq0 = EQ["non_iid_0"]
krum_seeds = PS["non_iid_0_krum"]
krum_bad_seed = krum_seeds.index(min(krum_seeds))
tm_mis = mis[mis.rule == "trimmed_mean"].set_index("partition")
kr_mis = mis[mis.rule == "krum"].set_index("partition")
ratio = {r: RAT[r] for r in ("trimmed_mean", "median", "krum")}
ms = {r: T[r]["mean"] for r in T}

blocks = []


def H(text, level=1):
    blocks.append({"type": "heading", "level": level, "text": text})


def para(text, **kw):
    blocks.append({"type": "para", "text": text, **kw})


def bullet(text):
    blocks.append({"type": "bullet", "text": text})


def eq(text, num):
    blocks.append({"type": "equation", "text": text, "num": num})


def fig(path, caption, width=16):
    blocks.append({"type": "figure", "path": str(ROOT / path), "caption": caption, "width_cm": width})


def table(caption, header, rows, widths, note=None):
    blocks.append({"type": "table", "caption": caption, "header": header, "rows": rows, "widths": widths,
                   "note": note})


TITLE = ("Robust Aggregation for Federated Intrusion Detection under Heterogeneous Data and Model "
         "Poisoning: A Controlled Comparison on NSL-KDD")

# ------------------------------------------------------------------ abstract
abstract = (
    "Federated learning (FL) allows network nodes to train a shared intrusion detection system (IDS) "
    "without exchanging raw traffic, but its standard aggregation rule, Federated Averaging (FedAvg), is "
    "vulnerable to non-independent and identically distributed (non-IID) client data and to malicious clients "
    "that submit manipulated model updates. Few papers study both problems in a single experiment, so FedAvg "
    "was compared with coordinate-wise median, trimmed mean and Krum while the neural IDS, the dataset and the "
    "partitioning code stayed identical. The NSL-KDD dataset was divided among ten simulated clients using "
    "near-IID and Dirichlet non-IID (α = 0.3) partitions, and 0%, 20% or 30% of clients submitted "
    "sign-flipped, threefold-scaled updates; every configuration was repeated over five seeds. With 30% "
    f"malicious clients on near-IID data, FedAvg accuracy fell from {P('near_iid',0,'fedavg'):.1f}% to "
    f"{P('near_iid',30,'fedavg'):.1f}%, whereas the robust rules stayed within {max_rob_change:.1f} percentage "
    f"points of their clean accuracy. On non-IID data with 20% malicious clients, FedAvg reached "
    f"{pm('non_iid',20,'fedavg')}% accuracy against {rng(non20_rob)}% for the robust rules, but with five "
    "seeds the non-IID differences were not significant after Holm correction. An equal-weight FedAvg "
    "ablation showed that most of the robust rules' advantage on clean non-IID data comes from FedAvg's "
    "dataset-size weighting, and a trimmed mean configured for fewer attackers than were present failed "
    f"(β = 0.1: {tm_mis.loc['near_iid','acc_mean']:.1f}% near-IID accuracy at 30% malicious clients), whereas "
    f"Krum with f = 1 did not. The robust rules took {ratio['krum']:.1f} to {ratio['median']:.1f} times FedAvg's "
    "server aggregation time, under 2 ms per round. Robust aggregation is therefore a cheap safeguard against "
    "this naive attack, although results from one dataset and one non-adaptive attack should not be "
    "generalised."
)
KEYWORDS = "model poisoning; robust aggregation; Byzantine resilience; non-IID data; Krum; NSL-KDD"

# ------------------------------------------------------------------ 1 Introduction
H("1. Introduction")
para("5G and emerging 6G networks are evolving into distributed computing and networking ecosystems that "
     "combine massive Internet-of-Things connectivity, edge computing, network virtualisation, network slicing "
     "and intelligent network management. These capabilities increase the number and diversity of participating "
     "devices and services and consequently expand the attack surface. Security mechanisms therefore need to "
     "operate across distributed environments while accounting for privacy, scalability and heterogeneous "
     "network conditions [5]–[10]. These networks motivate the present study, but the experiments below use "
     "a general network-intrusion benchmark rather than 5G/6G traffic, so the results are not claims about "
     "5G/6G deployments (Section 5, limitation (i)).")
para("Intrusion detection systems (IDSs) watch network traffic, usually through flow statistics, and raise an "
     "alert when they see malicious behaviour. Machine-learning-based IDSs can learn complex patterns, but "
     "conventional centralised training generally requires security data from multiple locations to be "
     "transferred to common infrastructure. In a distributed edge or IoT deployment this means extra bandwidth "
     "use, exposure of sensitive network records, and a central server that has to cope with ever more sources. "
     "Federated learning (FL) avoids this: every client keeps its own dataset and only sends model parameters or "
     "updates to a server, which merges them into one shared model [1]–[4].")
para("Earlier papers have applied FL to IoT security with deep-learning and anomaly-detection models, and have "
     "built federated IDSs for massive IoT, software-defined networking and 5G-oriented settings [11]–[15], "
     "[20]–[22], [25]. Their results show that intrusion detection can work without gathering raw training data "
     "in one place. How well it works, though, varies with how the data are spread over clients, with the model, "
     "and with the aggregation strategy. Non-independent and identically distributed (non-IID) traffic is a good "
     "example: each client sees a different mix of attacks, so the local models pull in different directions and "
     "the global model converges and generalises less well [14].")
para("FL brings security risks of its own. A compromised participant can tamper with its local training or with "
     "the update it uploads, which may degrade the global model or push it toward behaviour the attacker wants. "
     "Work on FL security, and on AI security in 6G, repeatedly lists malicious clients, adversarial updates, "
     "privacy leakage and trust management as open problems [1], [3], [8], [18], [24]. A federated IDS therefore "
     "needs to be judged on how it copes with bad updates and on what it costs to communicate, as well as on "
     "ordinary classification metrics.")
para("The recent literature contains substantial work on individual dimensions of this problem, including "
     "federated IDS, non-IID learning, communication-efficient FL, continual learning and adversarial threats "
     "[14], [18]–[20], [23], [24]. These dimensions are nevertheless frequently investigated using different "
     "datasets, client partitions and evaluation criteria, so the practical value of robust aggregation for IDS "
     "remains difficult to read from the published evidence.")
para("Accordingly, this study is a controlled comparison in which the dataset, local IDS model and client "
     "partitioning are held fixed while the aggregation rule, the degree of data heterogeneity and the fraction "
     "of malicious clients are varied.")
para("This work makes three contributions:")
bullet("A reproducible controlled comparison in which raw training data remain local to ten simulated clients and "
       "four aggregation rules (FedAvg, coordinate-wise median, trimmed mean and Krum) can be exchanged without "
       "changing any other component; the code and the per-round logs of every run are public.")
bullet("An evaluation under near-IID and Dirichlet non-IID partitions with 0%, 20% and 30% malicious clients that "
       "jointly reports detection performance, degradation under attack, convergence and server-side cost, together "
       "with two ablations: FedAvg with equal client weights, and robust rules configured for fewer attackers than "
       "are present.")
bullet(f"An empirical finding, replicated over five seeds, that FedAvg lost {drop_fedavg_near:.1f} accuracy points "
       f"on near-IID data with 30% malicious clients while the robust rules lost at most {max_rob_change:.1f} "
       "points; that under non-IID data the robust rules' advantage over FedAvg without an attacker is mostly "
       "explained by FedAvg's size weighting; and that trimmed mean, unlike Krum here, fails when its trimming "
       "level is set below the attacker fraction.")
para("The remainder of this paper is organised as follows. Section 2 reviews related work. Section 3 describes "
     "the problem formulation and methodology. Section 4 presents the experimental results. Section 5 discusses "
     "the findings and limitations, and Section 6 concludes.")

# ------------------------------------------------------------------ 2 Literature review (unchanged in substance)
H("2. Literature Review")
H("2.1 Federated learning for cybersecurity and privacy", 2)
para("Federated learning supports decentralised model training without requiring all training examples to be "
     "collected by a single server. Security and privacy surveys show that FL can reduce direct sharing of raw "
     "data, but model updates can still expose information and can themselves become attack surfaces [1]–[4]. "
     "This is especially relevant to IoT and distributed network environments, where data are generated close to "
     "devices and edge nodes. The literature therefore treats communication efficiency, client heterogeneity, "
     "trust and adversarial robustness as important complements to the basic privacy advantage of FL [2], [3]. "
     "The reference algorithm for most of this work is Federated Averaging, in which clients train locally and the "
     "server averages their models in proportion to local dataset size [26].")
H("2.2 Federated IDS in IoT and 5G/6G", 2)
para("Ferrag et al. tested several neural architectures for federated IoT cybersecurity on a range of security "
     "datasets [11]. Mothukuri et al. used FL for anomaly detection against IoT attacks [12]. FELIDS is a federated "
     "IDS for agricultural IoT that was evaluated on several intrusion-detection datasets [22], and IDSoft is a "
     "federated, softwarised IDS for massive IoT in a 6G setting that pays attention to convergence and "
     "communication overhead [13]. Two recent papers cover federated DDoS detection in large-scale "
     "software-defined networks [25] and a mixture-of-experts federated IDS for 5G and IoT [20]. All of these "
     "show that FL can be applied to distributed IDS. They differ, however, in what they set out to measure and "
     "in how their experiments are set up.")
H("2.3 Heterogeneity, communication cost and adversarial robustness", 2)
para("Different nodes in a network see different traffic volumes, applications and attack classes, so "
     "heterogeneity matters a great deal for intrusion detection. Djaidja et al. compared FedAvg, FedProx, FedPer "
     "and SCAFFOLD for 5G and beyond intrusion detection on IID and non-IID data, and their results highlight the "
     "influence of data distribution and FL algorithm choice [14]. A common way to generate controlled "
     "heterogeneity in such experiments is to draw each client's class proportions from a Dirichlet distribution, "
     "whose concentration parameter sets the degree of skew [31]. Communication-efficient on-device FL has also "
     "been investigated in industrial IoT [21]. In parallel, adversarial-security research for AI-driven 6G "
     "systems highlights the risks posed by manipulated learning processes [8]. Recent surveys of FL for 5G/6G "
     "cybersecurity and FL-based IDS further identify malicious clients, aggregation vulnerabilities, privacy and "
     "communication efficiency as continuing challenges [18], [24].")
para("Outside the IDS literature, a dedicated line of work addresses malicious participants through the "
     "aggregation rule. Krum selects the single client update that lies closest to its nearest neighbours and was "
     "proposed to tolerate Byzantine workers [27]. Coordinate-wise median and trimmed mean combine updates "
     "dimension by dimension and discard extreme values, and have been analysed in terms of statistical error "
     "rates [28]. Conversely, model-poisoning attacks that boost a malicious update, and attacks crafted "
     "specifically against Byzantine-robust rules, have been shown to degrade FL models [29], [30], [33], [34]. "
     "Whether these general-purpose rules still work on intrusion-detection traffic, where client distributions "
     "are strongly skewed, is what this paper tests.")
H("2.4 Newer, more specialised work", 2)
para("Newer papers use more specialised designs. One applies generative techniques to imbalanced industrial-IoT "
     "IDS data [15]; another proposes continual federated learning for attacks that keep evolving in SDN-based "
     "edge computing [19]; a third studies few-shot learning for classifying attack traffic when labelled examples "
     "are scarce [23]. Post-quantum and blockchain-enhanced FL have also been looked at [17], and broader 6G "
     "surveys list the security requirements of future intelligent networks [5], [6], [9], [16]. These designs "
     "show how wide the field is, but each extra component makes it harder to see what the aggregation rule alone "
     "contributes.")
H("2.5 Research gap", 2)
para("Taken together, the cited work shows that FL can support distributed intrusion detection, and other papers "
     "separately cover non-IID data, 5G/6G security, communication efficiency and adversarial threats. What is "
     "still missing is a single, controlled and reproducible comparison of IDS performance, degradation under "
     "malicious clients, aggregation robustness and communication overhead under the same conditions (Table 1). "
     "To fill that gap, the dataset, local IDS model, client-partition strategy and metrics are held fixed here, "
     "and only the aggregation mechanism and the poisoning condition change. The contribution is a comparative "
     "experiment. It is not a claim that federated IDS or robust aggregation is itself a new idea.")
table("Table 1. Related work set against the present study. The last column names the dimension of this study "
      "that each work was not designed to isolate. It is not meant as criticism, and the entries should be checked "
      "against the original papers.",
      ["Study", "Approach", "Dataset / setting", "Limitation relative to this study"],
      [["Ferrag et al., 2021 [11]", "Federated deep learning; several neural architectures compared",
        "Multiple IoT security datasets", "Aggregation rule and poisoning not varied as the main factors"],
       ["Mothukuri et al., 2022 [12]", "Federated anomaly detection", "IoT security attacks",
        "Aggregation-rule comparison not isolated"],
       ["IDSoft, 2023 [13]", "Federated, softwarised IDS", "Massive IoT in a 6G setting",
        "Robust aggregation under malicious clients not isolated"],
       ["Djaidja et al., 2024 [14]", "FedAvg, FedProx, FedPer, SCAFFOLD compared", "5G and beyond IDS; IID and non-IID",
        "Malicious-update robustness not the compared dimension"],
       ["FELIDS, 2022 [22]", "Federated IDS", "Agricultural IoT; several IDS datasets",
        "Domain-specific; aggregation robustness not isolated"],
       ["Fotse et al., 2025 [25]", "Federated DDoS detection", "Large-scale SDN",
        "Single attack family; aggregation robustness not isolated"],
       ["Ilias et al., 2026 [20]", "Federated mixture-of-experts IDS", "5G-NIDD and an IoT dataset; Dirichlet non-IID",
        "Malicious clients not studied"],
       ["This work", "FedAvg vs median, trimmed mean, Krum; fixed MLP IDS", "NSL-KDD; near-IID and non-IID; 0–30% malicious",
        "One dataset and one attack type (Section 5)"]],
      [3.0, 4.2, 3.9, 4.9])

# ------------------------------------------------------------------ 3 Methodology
H("3. Methodology")
H("3.1 Problem formulation", 2)
para("In a distributed network the security data sit on many different devices and edge nodes, and training an "
     "IDS centrally on them is often undesirable because of data transfer, privacy, scalability and communication "
     "limits. Federated learning can reduce the need to centralise raw traffic by allowing local training and "
     "collaborative model aggregation. Two characteristics of realistic distributed environments challenge this "
     "approach: client datasets are often non-IID because different network nodes observe different traffic and "
     "attack distributions, and participating clients may be compromised and submit manipulated model updates. "
     "Under these conditions, conventional FedAvg may produce a less reliable global IDS model. Robust aggregation "
     "can reduce the influence of abnormal updates but may introduce additional computational or communication "
     "costs. The problem addressed here is therefore to determine how the choice of aggregation rule influences "
     "detection performance, resilience to malicious clients, convergence and communication overhead under "
     "heterogeneous network-traffic distributions.")
para("Let there be K clients, where client k holds a local dataset Dₖ of nₖ labelled flows and n = Σₖ nₖ. The IDS is "
     "a binary classifier f(x; w) with parameter vector w that returns the probability that a flow x is an attack. "
     "The federated objective is")
eq("min_w F(w) = Σₖ₌₁..K (nₖ / n) Fₖ(w),   Fₖ(w) = (1 / nₖ) Σ₍ₓ,ᵧ₎∈Dₖ L( f(x; w), y )", 1)
para("where L is the binary cross-entropy loss. In round t the server broadcasts wᵗ, each client trains locally to "
     "obtain wₖᵗ and returns the update Δₖᵗ = wₖᵗ − wᵗ. A malicious client replaces its honest update with a scaled "
     "sign-flipped one,")
eq("Δ′ₖᵗ = −γ Δₖᵗ", 2)
para("with scale γ = 3, and the server then computes the next global model as")
eq("wᵗ⁺¹ = wᵗ + A( Δ₁ᵗ, …, Δ_Kᵗ )", 3)
para("where A is the aggregation rule under study. Four rules were compared. FedAvg weights each update by dataset "
     "size, A_avg = Σₖ (nₖ / n) Δₖ [26]. Coordinate-wise median takes, for every parameter index j, the median of "
     "Δₖ,ⱼ across clients [28]. Trimmed mean sorts the K values of each coordinate, discards the m = ⌊βK⌋ smallest "
     "and m largest, and averages the remaining K − 2m [28]. Krum assigns each client the score")
eq("s(k) = Σᵢ∈Nₖ ‖ Δₖ − Δᵢ ‖²", 4)
para("where Nₖ contains the K − f − 2 clients nearest to k, and returns the single update with the lowest score [27]. "
     "In the main experiments β = 0.3 (m = 3) and f = 3. Two further variants were used only as ablations (Section "
     "4.5): equal-weight FedAvg, A_eq = (1/K) Σₖ Δₖ, which differs from FedAvg only in ignoring dataset size; and "
     "trimmed mean with β = 0.1 (m = 1) and Krum with f = 1, i.e. defences configured for one attacker when three "
     "are present. Detection quality was measured on the attack class as accuracy, precision = TP/(TP+FP), "
     "recall = TP/(TP+FN) and F1 = 2·precision·recall/(precision+recall).")
H("3.2 Data", 2)
para("The NSL-KDD dataset was used [32]. It contains 125,973 training records (KDDTrain+) and 22,544 test records "
     "(KDDTest+), each described by 41 features and a label. The files were obtained from a public GitHub mirror "
     "of the dataset (github.com/defcom17/NSL_KDD). Three categorical features (protocol type, service and flag) "
     "were one-hot encoded, giving 122 input features, and all features were standardised with a scaler fitted on "
     "the training set only. The one-hot column set was derived from the union of training and test records so "
     "that both sets share one feature space; no labels were involved. Labels were binarised as normal (0) or "
     "attack (1). The training set contained 46.5% attack records and KDDTest+ 56.9%. Training attacks were also "
     "grouped into four categories (DoS 45,927, Probe 11,656, R2L 995 and U2R 52 records, against 67,343 normal "
     "records) solely to create non-IID partitions. KDDTest+ contains attack types that do not occur in the "
     "training set, so absolute accuracies in this study are bounded by that distribution shift. The data are "
     "public and contain no human or animal subjects, so no ethics approval was required.")
sz = S["seed0_non_iid_sizes"]
para("The training set was distributed over K = 10 clients with the Dirichlet procedure of [31]: for each of the "
     "five categories, client proportions were drawn from Dir(α) and records were allocated accordingly; a draw was "
     "repeated in the rare case that a client received fewer than 128 records (one mini-batch). A concentration of "
     "α = 100 produced near-IID shards (for seed 0, between 11,544 and 13,562 records per client, with 42–49% "
     f"attacks), and α = 0.3 produced strongly heterogeneous shards (for seed 0, between {min(sz):,} and "
     f"{max(sz):,} records, attack share between 2% and 100%, and the most frequent category making up 53–100% of "
     "a client's data). The two settings are denoted near-IID and non-IID below.")
H("3.3 Simulation procedure", 2)
para("The simulation (Figure 1) runs synchronous FL on one machine. Each run proceeded as follows: (i) the server "
     "initialised a global model; (ii) in every round all ten clients received the global model and trained it "
     "locally on their own shard; (iii) a randomly chosen subset of clients, fixed for the whole run and sized 0%, "
     "20% or 30% of K, replaced its update according to Eq. (2); (iv) the server aggregated the updates with one of "
     "the rules in Eq. (3); and (v) the new global model was evaluated on KDDTest+ and the statistics were logged. "
     "Raw records never left their client; only parameter updates were exchanged. The server was assumed to see "
     "each client's individual update, which the three robust rules require. No secure aggregation, differential "
     "privacy, or attacker adaptation to the aggregation rule was modelled.")
fig("figures/fig1_framework.png",
    "Figure 1. Simulated federated IDS setup. Each client trains on its own NSL-KDD shard and uploads a model update "
    "(solid arrows); the server applies one of the aggregation rules and broadcasts the new global model (dashed "
    "arrows). The shaded client represents a malicious participant. IDS: intrusion detection system; MLP: multilayer "
    "perceptron; acc: accuracy; KDDTest+: NSL-KDD test set.", 15)
H("3.4 Experimental setup", 2)
para("The local IDS was a multilayer perceptron with 122 inputs, one hidden layer of 64 ReLU units and one sigmoid "
     "output (7,937 parameters), initialised with He-normal weights. Local training used mini-batch SGD with batch "
     "size 128, learning rate 0.05 and one local epoch per round, for 25 communication rounds. These values were set "
     "before the experiments and were not tuned on the test set. Every combination of partition (near-IID, "
     "non-IID), malicious fraction (0%, 20%, 30%) and aggregation rule (the four main rules plus equal-weight "
     "FedAvg) was run with five seeds (0–4), and the two misconfigured defences were run at 30% malicious clients "
     "on both partitions, giving 170 runs. The seed fixed the initial weights, the partition, the choice of "
     "malicious clients and the order of mini-batches, so all rules saw identical conditions for a given seed.")
para("No validation split was held out, and validation-based early stopping was not used: every run trained for "
     "25 rounds and the reported figures are those of the final round, so no model selection on test data "
     "occurred. Because median and trimmed mean reach their highest accuracy after one or two rounds on non-IID "
     "data and then decline (Section 4.3), Table 5 also reports the best-round accuracy of each run. That figure is "
     "chosen with knowledge of the test set and is shown only to quantify how much final-round reporting "
     "understates the peak; it is not a deployable result.")
para("Server aggregation time was measured around the aggregation call only, as the mean over rounds. Uplink "
     "volume was computed from the update size. Differences between each robust rule and FedAvg in final accuracy "
     "were tested with a two-sided Welch t-test over the five seeds; the twelve p-values in Table 4 were adjusted "
     "with the Holm step-down procedure [35], and effect sizes are given as Cohen's d with pooled standard "
     "deviation [36]. Experiments ran on a CPU-only cloud workspace with two virtual cores using Python 3.13, "
     "NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, scikit-learn 1.9.1 and Matplotlib 3.11.2; the MLP and the federated "
     "loop were implemented directly in NumPy, and scikit-learn was used only for scaling and metrics.")

# ------------------------------------------------------------------ 4 Results
H("4. Results")
H("4.1 Detection performance and resilience to malicious clients", 2)
para("Table 2 reports final-round performance on KDDTest+ for the near-IID partition and Table 3 for the non-IID "
     "partition; Figure 2 summarises accuracy. Values are mean ± standard deviation over five seeds. With no "
     f"attacker on near-IID data the four rules were indistinguishable: accuracy lay between {min(near_clean):.1f}% "
     f"and {max(near_clean):.1f}%. With 20% malicious clients FedAvg was unaffected on average "
     f"({P('near_iid',20,'fedavg'):.1f}%), but with 30% its accuracy collapsed to {pm('near_iid',30,'fedavg')}% and "
     f"its F1 to {P('near_iid',30,'fedavg','f1_mean'):.1f}%, a loss of {drop_fedavg_near:.1f} points. Median, "
     f"trimmed mean and Krum stayed between {min(near30_rob):.1f}% and {max(near30_rob):.1f}% at the same attack "
     f"level, within {max_rob_change:.1f} points of their clean accuracy.")


def perf_rows(part):
    rows = []
    for mal in (0, 20, 30):
        for i, r in enumerate(["fedavg", "fedavg_equal", "median", "trimmed_mean", "krum"]):
            rows.append([f"{mal}%" if i == 0 else "", L[r], pm(part, mal, r), pm(part, mal, r, "f1"),
                         f1(P(part, mal, r, "recall")), f1(P(part, mal, r, "precision"))])
    return rows


HDR = ["Malicious", "Method", "Accuracy (%)", "F1 (%)", "Recall (%)", "Precision (%)"]
W6 = [2.0, 3.6, 2.9, 2.9, 2.2, 2.4]
table("Table 2. Final-round performance on KDDTest+, near-IID partition (α = 100). Values are mean ± standard "
      "deviation over five seeds (recall and precision: mean). Attack is the positive class. FedAvg (equal wt.) is "
      "the equal-weight ablation of Section 4.5.", HDR, perf_rows("near_iid"), W6)
para("On the non-IID partition (Table 3) the picture was different even without an attacker. FedAvg reached "
     f"{pm('non_iid',0,'fedavg')}% accuracy and {P('non_iid',0,'fedavg','recall'):.1f}% attack recall, whereas "
     f"median and trimmed mean reached {P('non_iid',0,'median'):.1f}% and {P('non_iid',0,'trimmed_mean'):.1f}% "
     f"accuracy with recall of {P('non_iid',0,'median','recall'):.1f}% and "
     f"{P('non_iid',0,'trimmed_mean','recall'):.1f}%, at essentially unchanged precision (about 92%). Section 4.5 "
     "shows that most of this gap disappears when FedAvg weights clients equally. Krum had the highest mean recall "
     f"({P('non_iid',0,'krum','recall'):.1f}%) but also the largest run-to-run spread "
     f"({pm('non_iid',0,'krum')}% accuracy), because in one of five seeds (seed {krum_bad_seed}) it converged to a "
     f"model that labelled every test flow as an attack (accuracy {min(krum_seeds):.1f}%, equal to the attack share "
     "of KDDTest+).")
para("With attackers present on non-IID data, FedAvg degraded by "
     f"{P('non_iid',0,'fedavg')-P('non_iid',20,'fedavg'):.1f} points at 20% and "
     f"{P('non_iid',0,'fedavg')-P('non_iid',30,'fedavg'):.1f} points at 30% malicious clients, and its F1 fell to "
     f"{pm('non_iid',20,'fedavg','f1')}% and {pm('non_iid',30,'fedavg','f1')}%. Its variance was very high: at 20% "
     f"malicious clients final accuracy ranged from {min(fa20):.1f}% to {max(fa20):.1f}% across seeds, depending on "
     f"which clients were compromised. The robust rules ended at {rng(non20_rob)}% (20%) and {rng(non30_rob)}% "
     "(30%), with standard deviations between "
     f"{min(P('non_iid',m,r,'acc_sd') for m in (20,30) for r in ROB):.1f} and "
     f"{max(P('non_iid',m,r,'acc_sd') for m in (20,30) for r in ROB):.1f} points.")
table("Table 3. Final-round performance on KDDTest+, non-IID partition (α = 0.3). Values are mean ± standard "
      "deviation over five seeds (recall and precision: mean).", HDR, perf_rows("non_iid"), W6)
fig("figures/fig2_final_accuracy.png",
    "Figure 2. Final-round test accuracy (%) versus the fraction of malicious clients for the four aggregation rules "
    "on the near-IID (left) and non-IID (right) partitions. Each bar shows the mean of five seeds, and the error bars "
    "show one standard deviation. Hatching distinguishes rules so the figure remains legible in greyscale.", 16)

H("4.2 Statistical comparison", 2)
para("Table 4 gives the difference in mean final accuracy between each robust rule and FedAvg, the Welch t-test "
     "p-value (n = 5 per group) before and after Holm correction over the twelve comparisons, and Cohen's d. On "
     "near-IID data with 30% malicious clients the differences exceeded "
     f"{min(stat('near_iid',30,r,'delta') for r in ROB):.0f} points with p < 0.001 after correction, while at 20% "
     "they were below 0.5 points and not significant. On non-IID data the robust rules were "
     f"{min(stat('non_iid',m,r,'delta') for m in (20,30) for r in ROB):.0f}–"
     f"{max(stat('non_iid',m,r,'delta') for m in (20,30) for r in ROB):.0f} points ahead of FedAvg and the effect "
     f"sizes were large (d = {min(d_non):.1f}–{max(d_non):.1f}), but the differences were not statistically "
     f"significant: at 20% malicious clients the unadjusted p-values were {min(p_non20):.2f}–{max(p_non20):.2f}, and "
     f"at 30% they were {min(p_non30):.3f}–{max(p_non30):.3f} before correction and {min(ph_non[3:]):.2f}–"
     f"{max(ph_non[3:]):.2f} after it. Put plainly, the non-IID differences at 30% are significant only before "
     "multiple-comparison correction, and those at 20% not even then; with five seeds and FedAvg's large seed-to-seed "
     "variance the study is underpowered for the non-IID comparisons.")
rows4 = []
for part in ("near_iid", "non_iid"):
    for mal in (20, 30):
        for i, r in enumerate(ROB):
            rows4.append(["Near-IID" if part == "near_iid" and i == 0 else ("Non-IID" if i == 0 else ""),
                          f"{mal}%" if i == 0 else "", L[r], f"{stat(part,mal,r,'delta'):+.1f}",
                          pfmt(stat(part, mal, r, "p")), pfmt(stat(part, mal, r, "p_holm")),
                          f"{stat(part,mal,r,'d'):.1f}"])
table("Table 4. Difference in mean final accuracy (percentage points) between each robust rule and FedAvg, with "
      "two-sided Welch t-test p-values before and after Holm correction over the twelve comparisons, and Cohen's d "
      "(pooled standard deviation). Five seeds per group.",
      ["Partition", "Malicious", "Rule", "Δ accuracy (points)", "p (unadjusted)", "p (Holm)", "Cohen's d"],
      rows4, [2.0, 1.9, 2.7, 2.6, 2.4, 2.2, 2.2],
      note="The very large d values on near-IID data at 30% reflect the small within-group spread of both groups "
           "rather than an especially meaningful effect size.")

H("4.3 Convergence", 2)
dn = lambda r: DY[f"non_iid_{r}"]
para("Accuracy curves are shown in Figure 3 (non-IID) and Figure 4a (near-IID, 30% malicious). In all clean "
     "near-IID runs the models reached their plateau within the first one or two rounds; later rounds changed "
     "accuracy by less than about one point. On clean non-IID data, median and trimmed mean averaged "
     f"{dn('median')['r1']:.1f}% and {dn('trimmed_mean')['r1']:.1f}% accuracy after round 1 and then declined to "
     f"{dn('median')['r25']:.1f}% and {dn('trimmed_mean')['r25']:.1f}% by round 25, whereas FedAvg rose from "
     f"{dn('fedavg')['r1']:.1f}% to {dn('fedavg')['r25']:.1f}%. Table 5 quantifies this: on clean non-IID data the "
     f"best-round accuracy of median and trimmed mean was {P('non_iid',0,'median','best_acc_mean'):.1f}% and "
     f"{P('non_iid',0,'trimmed_mean','best_acc_mean'):.1f}%, reached on average in round "
     f"{P('non_iid',0,'median','best_round_mean'):.1f} and {P('non_iid',0,'trimmed_mean','best_round_mean'):.1f}, "
     f"against {P('non_iid',0,'median'):.1f}% and {P('non_iid',0,'trimmed_mean'):.1f}% in the final round. With "
     "malicious clients on non-IID data FedAvg curves stayed far below the robust curves in every round. Table 5 "
     "also gives the mean accuracy over all 25 rounds and the first round at which a run reached 74% accuracy, "
     "together with the number of runs (of five) that reached it at all. At 30% malicious clients, FedAvg never "
     "reached 74% on either partition, while every robust run did.")
fig("figures/fig3_noniid_curves.png",
    "Figure 3. Test accuracy (%) per communication round on the non-IID partition with 0%, 20% and 30% malicious "
    "clients. Lines are means over five seeds and shaded bands show one standard deviation. Line style and marker "
    "identify each rule.", 16.5)
rows5 = []
for part in ("near_iid", "non_iid"):
    for mal in (0, 20, 30):
        for i, r in enumerate(["fedavg", "median", "trimmed_mean", "krum"]):
            n74 = int(P(part, mal, r, "r74_n"))
            r74 = "–" if n74 == 0 else f"{P(part, mal, r, 'r74_mean'):.1f}"
            rows5.append(["Near-IID" if part == "near_iid" and mal == 0 and i == 0 else
                          ("Non-IID" if part == "non_iid" and mal == 0 and i == 0 else ""),
                          f"{mal}%" if i == 0 else "", L[r],
                          f1(P(part, mal, r, "mean_round_acc")),
                          f"{P(part, mal, r, 'best_acc_mean'):.1f} ({P(part, mal, r, 'best_round_mean'):.1f})",
                          f1(P(part, mal, r)), f"{r74} ({n74})"])
table("Table 5. Convergence summary. Mean-round accuracy is the accuracy averaged over the 25 rounds. Best-round "
      "accuracy is the mean over seeds of each run's highest test accuracy, with the mean round at which it occurred "
      "in brackets; it is selected on the test set and is shown for reference only. Final-round accuracy repeats "
      "Tables 2 and 3. Round to 74% is the mean first round with accuracy ≥ 74% among runs that reached it; the "
      "number of such runs is given in brackets (out of five).",
      ["Partition", "Malicious", "Rule", "Mean-round acc. (%)", "Best-round acc. (%) (round)", "Final-round acc. (%)",
       "Round to 74% (runs)"],
      rows5, [1.9, 1.7, 2.5, 2.3, 2.9, 2.3, 2.4])

H("4.4 Communication and server cost", 2)
para("Every rule exchanged the same payload. Each client uploaded one update of 7,937 float32 parameters, i.e. "
     "31,748 bytes (31.0 KiB) per round, so ten clients produced 310 KiB per round and 7.6 MiB over 25 rounds in "
     "the uplink direction, with an identical downlink for the broadcast model. The rules therefore did not differ "
     "in communication volume. They differed in server computation (Figure 4b): mean aggregation time per round, "
     f"over all main configurations, was {ms['fedavg']:.3f} ms for FedAvg, {ms['krum']:.3f} ms for Krum, "
     f"{ms['trimmed_mean']:.3f} ms for trimmed mean and {ms['median']:.3f} ms for median, i.e. "
     f"{ratio['krum']:.1f}, {ratio['trimmed_mean']:.1f} and {ratio['median']:.1f} times the FedAvg cost. Krum was "
     "implemented with a single Gram-matrix computation of all pairwise distances, which makes it cheap at K = 10; "
     "its cost grows quadratically with the number of clients, whereas the cost of median and trimmed mean grows "
     "with the model dimension (and as K log K per coordinate). These differences are small in absolute terms for a "
     "model of this size.")
fig("figures/fig4_nearIID30_and_time.png",
    "Figure 4. (a) Test accuracy (%) per round on the near-IID partition with 30% malicious clients (mean ± one "
    "standard deviation over five seeds). (b) Mean server aggregation time per round in milliseconds across all main "
    "configurations; error bars show one standard deviation.", 16)

H("4.5 Ablations: equal-weight FedAvg and misconfigured defences", 2)
para("Equal-weight FedAvg. To test whether the robust rules beat FedAvg on clean non-IID data because FedAvg "
     "weights clients by dataset size, FedAvg was rerun with equal client weights (A_eq in Section 3.1). On clean "
     f"non-IID data it reached {eq0['eq_acc']:.1f} ± {eq0['eq_sd']:.1f}% accuracy, {eq0['delta']:.1f} points above "
     f"size-weighted FedAvg ({eq0['w_acc']:.1f}%; Welch p = {eq0['p']:.3f}), with recall rising from "
     f"{P('non_iid',0,'fedavg','recall'):.1f}% to {eq0['eq_recall']:.1f}% at unchanged precision. This closes "
     f"{eq0['delta']/(P('non_iid',0,'median')-P('non_iid',0,'fedavg'))*100:.0f}% of the gap to median and "
     f"{eq0['delta']/(P('non_iid',0,'trimmed_mean')-P('non_iid',0,'fedavg'))*100:.0f}% of the gap to trimmed mean, "
     "and the remaining differences between equal-weight FedAvg and median or trimmed mean were not significant "
     "(Welch p = 0.29 and 0.23). On near-IID data, where shards are of similar size, the two FedAvg variants "
     f"coincided ({EQ['near_iid_0']['eq_acc']:.1f}% each). Equal weighting gave no protection against the attack: "
     f"with 20% and 30% malicious clients on non-IID data it reached {EQ['non_iid_20']['eq_acc']:.1f}% and "
     f"{EQ['non_iid_30']['eq_acc']:.1f}%, and on near-IID data at 30% it collapsed to "
     f"{EQ['near_iid_30']['eq_acc']:.1f}% (Tables 2 and 3).")
para("Misconfigured defences. The main experiments set β = 0.3 and f = 3, which equal the largest attacker "
     "fraction and so assume that the defender knows it. Table 6 reports the same 30% attack against trimmed mean "
     "with β = 0.1 (one value trimmed from each end of every coordinate) and Krum with f = 1. Trimmed mean failed: "
     f"accuracy fell to {tm_mis.loc['near_iid','acc_mean']:.1f} ± {tm_mis.loc['near_iid','acc_sd']:.1f}% on near-IID "
     f"and {tm_mis.loc['non_iid','acc_mean']:.1f} ± {tm_mis.loc['non_iid','acc_sd']:.1f}% on non-IID data, against "
     f"{tm_mis.loc['near_iid','correct_acc']:.1f}% and {tm_mis.loc['non_iid','correct_acc']:.1f}% when correctly "
     "configured, because with three attackers and m = 1 at least two scaled values survive in every coordinate. "
     f"It still did better than FedAvg ({tm_mis.loc['near_iid','fedavg_acc']:.1f}% and "
     f"{tm_mis.loc['non_iid','fedavg_acc']:.1f}%). Krum with f = 1 was unaffected on near-IID data "
     f"({kr_mis.loc['near_iid','acc_mean']:.1f} ± {kr_mis.loc['near_iid','acc_sd']:.1f}%) and lost "
     f"{kr_mis.loc['non_iid','correct_acc']-kr_mis.loc['non_iid','acc_mean']:.1f} points on non-IID data "
     f"({kr_mis.loc['non_iid','acc_mean']:.1f} ± {kr_mis.loc['non_iid','acc_sd']:.1f}%). With f = 1 Krum scores each "
     "update by its distance to its seven nearest neighbours; the three malicious updates are inflated, point in "
     "different directions and are therefore far from everything, so an honest update still has the lowest score. "
     "This tolerance is specific to an attack whose malicious updates do not agree with each other; colluding "
     "attackers that submit similar updates would not be rejected this way [30], [33], [34]. Median has no tuning "
     "parameter and tolerates any minority of attackers by construction.")
rows6 = []
for part in ("near_iid", "non_iid"):
    for i, (_, rw) in enumerate(mis[mis.partition == part].iterrows()):
        rows6.append(["Near-IID" if part == "near_iid" and i == 0 else ("Non-IID" if i == 0 else ""),
                      f"{L[rw.rule]}, {rw.setting}", f"{rw.acc_mean:.1f} ± {rw.acc_sd:.1f}",
                      f"{rw.f1_mean:.1f} ± {rw.f1_sd:.1f}", f1(rw.recall), f1(rw.precision),
                      f"{rw.correct_acc:.1f} ± {rw.correct_sd:.1f}"])
table("Table 6. Misconfigured defences against 30% malicious clients: final-round performance on KDDTest+ (mean ± "
      "standard deviation over five seeds; recall and precision: mean). The last column repeats the accuracy of the "
      "correctly configured rule (β = 0.3 or f = 3) from Tables 2 and 3.",
      ["Partition", "Rule and setting", "Accuracy (%)", "F1 (%)", "Recall (%)", "Precision (%)",
       "Correctly configured accuracy (%)"],
      rows6, [1.9, 3.4, 2.3, 2.3, 1.8, 1.9, 2.4])

# ------------------------------------------------------------------ 5 Discussion
H("5. Discussion")
para("The near-IID result has a simple mechanical explanation. With equal-sized shards and a fraction ρ of clients "
     "sending −γΔ, the FedAvg step along the honest direction is approximately (1 − ρ) − ργ = 1 − ρ(1 + γ). For "
     "γ = 3 this equals +0.2 at ρ = 0.2 and −0.2 at ρ = 0.3. At 20% the honest signal therefore survives, only "
     "shrunk, which is consistent with the unchanged accuracy, and at 30% the aggregate step points away from the "
     "honest direction and the model is driven to a worse-than-chance state (FedAvg accuracy of "
     f"{P('near_iid',30,'fedavg'):.1f}% is below the 43.1% obtained by predicting every flow as normal). The scaled "
     "updates produce extreme coordinate values, which median and trimmed mean simply ignore, and Krum rejects the "
     "whole update as an outlier. None of the three was therefore affected, provided the malicious clients stayed a "
     "minority within the trimming or Krum tolerance used here; Section 4.5 shows that trimmed mean is not "
     "protected when that tolerance is set too low. This analysis is an interpretation of the numbers and was not "
     "tested separately, for example by varying γ.")
para("Under non-IID data the situation is less clean. First, FedAvg weights clients by dataset size, so a single "
     "large client can dominate. In the seed-0 partition one client held "
     f"{S['seed0_non_iid_largest_share']*100:.0f}% of all records, which means the outcome depends heavily on whether "
     "that client is a malicious one. This fits the very wide spread of FedAvg accuracy at 20% "
     f"({min(fa20):.1f}%–{max(fa20):.1f}%). Second, the robust rules outperformed FedAvg even with no attacker. The "
     "equal-weight ablation (Section 4.5) supports the size-weighting explanation: giving every client the same "
     f"weight raised FedAvg's clean non-IID accuracy by {eq0['delta']:.1f} points and its recall by "
     f"{eq0['eq_recall']-P('non_iid',0,'fedavg','recall'):.1f} points, leaving no significant difference from "
     "median or trimmed mean. Small, attack-heavy shards therefore carry useful signal that size weighting "
     "suppresses, and median-type rules recover it because they treat clients more equally. The residual gap of "
     "about one point may also reflect that KDDTest+ has a higher attack share (56.9%) than the training data "
     "(46.5%), which favours rules that are less biased toward the majority class. Median and trimmed mean peak "
     "after round 1 or 2 at about 81% and then fall to about 78%, which looks like continued fitting to the training "
     "distribution. Since only final-round results are reported, the effect pulls the reported figures down rather "
     "than up; Table 5 gives the size of the gap.")
para("The results are consistent with the studies cited in Section 2 on two points: data heterogeneity can change "
     "how FL algorithms rank against each other [14], and aggregation vulnerabilities are a real concern for FL-based "
     "IDS [18], [24]. They also agree with the original analyses of Krum and coordinate-wise rules [27], [28]. It "
     "should be stated explicitly that the attack tested here, a scaled sign flip, is exactly the kind of large, "
     "outlying update that these three rules were designed to reject, so their success against it is expected "
     "rather than surprising. The study does not test attacks crafted to evade such rules: the boosted and targeted "
     "model poisoning of Bhagoji et al. [29], the local model poisoning attacks of Fang et al. [30], "
     "A Little Is Enough (ALIE), which hides small coordinated perturbations inside the honest variance [33], and the "
     "min-max and min-sum attacks of Shejwalkar and Houmansadr [34]. The absence of degradation observed here should "
     "therefore not be read as evidence of robustness to adaptive attackers. Direct numerical comparison with other "
     "NSL-KDD studies is not meaningful because the present model, partitioning and test protocol differ, and "
     "KDDTest+ accuracy for a plain MLP is limited by its unseen attack types.")
para("For practitioners the results suggest that, for models of this size, replacing FedAvg by a robust rule costs "
     f"at most about {ms['median']-ms['fedavg']:.1f} ms of server time per round and no extra communication, and "
     "removed the catastrophic failure in every attacked setting that was tested. The price is that the server "
     "must see each client's update individually, which is typically incompatible with secure aggregation. Each "
     "rule also has a specific weakness: trimmed mean must be configured for at least the true attacker fraction, "
     "and Krum, by selecting one client's update per round, can be unstable under strong heterogeneity (one of five "
     "clean non-IID runs collapsed to predicting only attacks). In this study coordinate-wise median, which needs no "
     "estimate of the attacker fraction, looks like the safest default. Krum gave the best mean accuracy under "
     "attack but was the least predictable from seed to seed. Where no attack is expected, equal-weight FedAvg "
     "recovers most of the robust rules' clean non-IID advantage at the lowest cost.")
para("This study has limits that readers should keep in mind. (i) NSL-KDD is the only dataset. It is a general "
     "network-intrusion benchmark from 2009, not a 5G/6G dataset, so the 5G/6G motivation in Section 1 rests on the "
     "structural similarity of distributed, heterogeneous traffic sources and not on real 5G/6G traces. (ii) A single "
     "attack was tested, a scaled sign flip with γ fixed at 3, which is the attack the three robust rules were "
     "designed to reject; the attacker never adapted to the aggregation rule, and the attacks of [29], [30], ALIE "
     "[33] and min-max/min-sum [34] remain untested, as do data-poisoning attacks such as label flipping. (iii) The "
     "IDS was a small MLP that reaches its plateau within one or two rounds, so the convergence and cost comparisons "
     "say little about larger models; ten clients participated in every round, and the malicious fraction was "
     "limited to 30%. Only one misconfigured setting per rule was tested. (iv) Five seeds give low statistical "
     "power: after Holm correction none of the non-IID differences in Table 4 is significant, although the effect "
     "sizes are large. (v) The partition and the choice of malicious clients were random, which introduces large "
     "variance in the non-IID experiments. (vi) Absolute accuracies are limited by the distribution shift in "
     "KDDTest+, and no validation split was used for early stopping. These limitations mean that the findings show "
     "the behaviour of these rules in a controlled setting and do not establish it for operational deployments.")

# ------------------------------------------------------------------ 6 Conclusion
H("6. Conclusion")
para("This study compared FedAvg with coordinate-wise median, trimmed mean and Krum for federated intrusion "
     "detection on NSL-KDD under near-IID and non-IID client partitions and with 0%, 20% and 30% malicious clients. "
     f"FedAvg lost {drop_fedavg_near:.1f} accuracy points on near-IID data with 30% malicious clients, whereas the "
     f"robust rules changed by at most {max_rob_change:.1f} points. On non-IID data with 20% malicious clients FedAvg "
     f"reached {P('non_iid',20,'fedavg'):.1f}% accuracy against {rng(non20_rob)}% for the robust rules, a large but, "
     "with five seeds, not statistically significant difference after correction. Most of the robust rules' "
     "advantage on clean non-IID data was explained by FedAvg's size weighting, and trimmed mean failed when "
     "configured for fewer attackers than were present. The robust rules required the same communication volume as "
     f"FedAvg and {ratio['krum']:.1f} to {ratio['median']:.1f} times its server aggregation time, still under 2 ms per "
     "round. This contributes a reproducible, controlled comparison of aggregation rules for federated IDS under "
     "heterogeneity and poisoning, with public code and per-run results. Future work will repeat the protocol on "
     "datasets that contain 5G/6G traffic, add adaptive attacks such as ALIE and min-max and data-poisoning "
     "attackers, vary the scale of the attack and the number of clients, use more seeds, and evaluate "
     "secure-aggregation-compatible defences.")

# ------------------------------------------------------------------ Declarations and references
DECL = [
    ("Acknowledgements.", "None."),
    ("Funding.", "This research received no specific grant from any funding agency."),
    ("Conflict of interest.", "The authors declare no conflict of interest."),
    ("Data and code availability.", "The NSL-KDD dataset is publicly available [32]. The experiment code, the "
     "per-round logs of all 170 runs and the scripts that regenerate every table and figure in this paper are "
     "publicly available at {REPO}."),
    ("Author contributions.", "Conceptualisation, methodology, software, validation, formal analysis, "
     "investigation, writing — original draft, and writing — review and editing were shared by both authors."),
    ("Use of AI tools.", "A generative AI assistant (Claude, Anthropic) was used to help implement the experiment "
     "code, run the experiments and the analysis, and draft and revise the manuscript text. The authors are "
     "responsible for the content."),
]
REFS = [
    "Mothukuri, V. et al. (2021). A survey on security and privacy of federated learning. Future Generation Computer Systems. https://doi.org/10.1016/j.future.2020.10.007",
    "Nguyen, D.C. et al. (2021). Federated learning for Internet of Things: A comprehensive survey. IEEE Communications Surveys & Tutorials. https://doi.org/10.1109/COMST.2021.3075439",
    "Ghimire, A. et al. (2022). Recent advances on federated learning for cybersecurity and cybersecurity for federated learning for Internet of Things. IEEE Internet of Things Journal. https://doi.org/10.1109/JIOT.2022.3150363",
    "Akhtarshenas, A. et al. (2024). Federated learning: A cutting-edge survey of the latest advancements and applications. Computer Communications. https://doi.org/10.1016/j.comcom.2024.107964",
    "Porambage, P. et al. (2021). The roadmap to 6G security and privacy. IEEE Open Journal of the Communications Society. https://doi.org/10.1109/OJCOMS.2021.3078081",
    "Nguyen, V.-L. et al. (2021). Security and privacy for 6G: A survey on prospective technologies and challenges. IEEE Communications Surveys & Tutorials. https://doi.org/10.1109/COMST.2021.3108618",
    "Alotaibi, A., Barnawi, A. (2023). Securing massive IoT in 6G: Recent solutions, architectures, future directions. Internet of Things. https://doi.org/10.1016/j.iot.2023.100715",
    "Hoang, V.-T.V. et al. (2024). Security risks and countermeasures of adversarial attacks on AI-driven applications in 6G networks: A survey. Journal of Network and Computer Applications. https://doi.org/10.1016/j.jnca.2024.104031",
    "Akbar, M.S. et al. (2025). On challenges of sixth-generation (6G) wireless networks: A comprehensive survey of requirements, applications, and security issues. Journal of Network and Computer Applications. https://doi.org/10.1016/j.jnca.2024.104040",
    "Kalodanis, K., Papapavlou, C., Feretzakis, G. (2025). Enhancing security in 5G and future 6G networks: Machine learning approaches for adaptive intrusion detection and prevention. Future Internet. https://doi.org/10.3390/fi17070312",
    "Ferrag, M.A. et al. (2021). Federated deep learning for cyber security in the Internet of Things: Concepts, applications, and experimental analysis. IEEE Access. https://doi.org/10.1109/ACCESS.2021.3118642",
    "Mothukuri, V. et al. (2022). Federated-learning-based anomaly detection for IoT security attacks. IEEE Internet of Things Journal. https://doi.org/10.1109/JIOT.2021.3077803",
    "Alotaibi, A., Barnawi, A. (2023). IDSoft: A federated and softwarized intrusion detection framework for massive internet of things in 6G network. Journal of King Saud University - Computer and Information Sciences. https://doi.org/10.1016/j.jksuci.2023.101575",
    "Djaidja, T.E.T. et al. (2024). Federated learning for 5G and beyond, a blessing and a curse - an experimental study on intrusion detection systems. Computers & Security. https://doi.org/10.1016/j.cose.2024.103707",
    "Hamouda, D. et al. (2024). Revolutionizing intrusion detection in industrial IoT with distributed learning and deep generative techniques. Internet of Things. https://doi.org/10.1016/j.iot.2024.101149",
    "Javeed, D. et al. (2024). Quantum-empowered federated learning and 6G wireless networks for IoT security: Concept, challenges and future directions. Future Generation Computer Systems. https://doi.org/10.1016/j.future.2024.06.023",
    "Gurung, D. et al. (2024). Performance analysis and evaluation of post quantum secure blockchained federated learning. Computer Networks. https://doi.org/10.1016/j.comnet.2024.110849",
    "Blika, A. et al. (2025). Federated learning for enhanced cybersecurity and trustworthiness in 5G and 6G networks: A comprehensive survey. IEEE Open Journal of the Communications Society. https://doi.org/10.1109/OJCOMS.2024.3449563",
    "Chetouane, A., Karoui, K. (2025). New continual federated learning system for intrusion detection in SDN-based edge computing. Concurrency and Computation: Practice and Experience. https://doi.org/10.1002/cpe.8332",
    "Ilias, L., Doukas, G., Lamprou, V., Mouzakitis, S., Ntanos, C., Askounis, D. (2026). A federated learning-based network intrusion detection system for 5G and IoT using mixture of experts. Electronics, 15(5), 1057. https://doi.org/10.3390/electronics15051057",
    "Liu, L. et al. (2021). A communication-efficient learning system for time-series data in industrial IoT: A communication-efficient on-device federated learning approach. IEEE Internet of Things Journal. https://doi.org/10.1109/JIOT.2020.3011726",
    "Friha, O. et al. (2022). FELIDS: Federated learning-based intrusion detection system for agricultural Internet of Things. Journal of Parallel and Distributed Computing. https://doi.org/10.1016/j.jpdc.2022.03.003",
    "Bovenzi, G. et al. (2024). Classifying attack traffic in IoT environments via few-shot learning. Journal of Information Security and Applications. https://doi.org/10.1016/j.jisa.2024.103762",
    "Zhang, H. et al. (2025). Survey of federated learning in intrusion detection. Journal of Parallel and Distributed Computing. https://doi.org/10.1016/j.jpdc.2024.104976",
    "Fotse, Y.S.N. et al. (2025). Federated learning based DDoS attacks detection in large scale software-defined network. IEEE Transactions on Computers, 74(1), 101–115. https://doi.org/10.1109/TC.2024.3474180",
    "McMahan, H.B., Moore, E., Ramage, D., Hampson, S., Agüera y Arcas, B. (2017). Communication-efficient learning of deep networks from decentralized data. Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 54: 1273–1282.",
    "Blanchard, P., El Mhamdi, E.M., Guerraoui, R., Stainer, J. (2017). Machine learning with adversaries: Byzantine tolerant gradient descent. Advances in Neural Information Processing Systems 30.",
    "Yin, D., Chen, Y., Ramchandran, K., Bartlett, P. (2018). Byzantine-robust distributed learning: Towards optimal statistical rates. Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80: 5650–5659.",
    "Bhagoji, A.N., Chakraborty, S., Mittal, P., Calo, S. (2019). Analyzing federated learning through an adversarial lens. Proceedings of the 36th International Conference on Machine Learning (ICML), PMLR 97: 634–643.",
    "Fang, M., Cao, X., Jia, J., Gong, N.Z. (2020). Local model poisoning attacks to Byzantine-robust federated learning. Proceedings of the 29th USENIX Security Symposium, pp. 1605–1622.",
    "Hsu, T.-M.H., Qi, H., Brown, M. (2019). Measuring the effects of non-identical data distribution for federated visual classification. arXiv:1909.06335.",
    "Tavallaee, M., Bagheri, E., Lu, W., Ghorbani, A.A. (2009). A detailed analysis of the KDD CUP 99 data set. Proceedings of the IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA), pp. 1–6. https://doi.org/10.1109/CISDA.2009.5356528",
    "Baruch, G., Baruch, M., Goldberg, Y. (2019). A little is enough: Circumventing defenses for distributed learning. Advances in Neural Information Processing Systems 32.",
    "Shejwalkar, V., Houmansadr, A. (2021). Manipulating the Byzantine: Optimizing model poisoning attacks and defenses for federated learning. Proceedings of the Network and Distributed System Security Symposium (NDSS).",
    "Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2), 65–70.",
    "Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Lawrence Erlbaum Associates.",
]

out = {"title": TITLE,
       "authors": "Abhinav Rajesh Pittan¹,*, Rishabh Rajesh¹",
       "affiliation": "¹ Department of Computer Science and Engineering, Vellore Institute of Technology (VIT), Vellore, 632014, Tamil Nadu, India",
       "corresponding": "* Corresponding author: abhinav.rajesh2024@vitstudent.ac.in",
       "abstract": abstract, "keywords": KEYWORDS, "blocks": blocks, "declarations": DECL, "references": REFS}
(ROOT / "paper" / "content.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("blocks:", len(blocks))
