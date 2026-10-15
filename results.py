"""Paper tables from logs/. Run: python results.py"""

import json
from pathlib import Path

import numpy as np

LOGS = Path(__file__).resolve().parent / "logs"
SEEDS = [53, 59, 61, 101, 103]
MAX_STEP = 20

CONDS = ["no_shift", "damping_x50", "density_x0.2", "density_x10"]
ARMS = ["frozen", "predlast", "predfirst", "alra"]

COND_LABEL = {
    "no_shift": "No shift",
    "damping_x50": r"Damping $50\times$",
    "density_x0.2": r"Density $0.2\times$",
    "density_x10": r"Density $10\times$",
}
ARM_LABEL = {
    "frozen": "Frozen",
    "predlast": r"\texttt{predlast+enclast}",
    "predfirst": r"\texttt{predfirst+enclast}",
    "alra": "ALRA (ours)",
}


def first_success_steps(cond, arm, seed):
    hits, episodes = {}, set()
    for line in open(f"{LOGS}/{cond}/{arm}/seed{seed}/per_iter_logs.json"):
        r = json.loads(line)
        episodes.add(r["sample_idx"])
        if r["success"]:
            i = r["sample_idx"]
            hits[i] = min(hits.get(i, r["step"]), r["step"])
    return hits, len(episodes)


def curve(cond, arm, seed):
    # episodes stop logging once they succeed, so success is cumulative
    hits, n = first_success_steps(cond, arm, seed)
    steps = np.array(sorted(hits.values()))
    return np.array([100 * (steps <= k).sum() / n for k in range(1, MAX_STEP + 1)])


def all_curves():
    return {(c, a): np.stack([curve(c, a, s) for s in SEEDS]) for c in CONDS for a in ARMS}


def paired_gain(cv, cond, arm, step=MAX_STEP):
    d = cv[(cond, arm)][:, step - 1] - cv[(cond, "frozen")][:, step - 1]
    return d.mean(), d.std(ddof=1) / np.sqrt(len(SEEDS))


def run_spread(cv, cond, arm, step=MAX_STEP):
    v = cv[(cond, arm)][:, step - 1]
    return v.mean(), v.std(ddof=1)


BOLD_MARGIN = 2.0


def bold_set(cv, cond):
    adapted = [a for a in ARMS if a != "frozen"]
    mean = {a: cv[(cond, a)][:, -1].mean() for a in adapted}
    best = max(adapted, key=mean.get)
    return {a for a in adapted if mean[best] - mean[a] <= BOLD_MARGIN}


def verify(cv):
    bad = []
    for c in CONDS:
        for a in ARMS:
            for i, s in enumerate(SEEDS):
                reported = None
                for line in open(f"{LOGS}/{c}/{a}/seed{s}/logs.json"):
                    d = json.loads(line)
                    if "final_eval/success_rate" in d:
                        reported = 100 * d["final_eval/success_rate"]
                _, n = first_success_steps(c, a, s)
                if n != 100:
                    bad.append((c, a, s, f"{n} episodes"))
                if reported is None or abs(reported - cv[(c, a)][i, -1]) > 1e-9:
                    bad.append((c, a, s, f"reported {reported} != derived {cv[(c, a)][i, -1]}"))
    return bad


def results_table(cv):
    rows = []
    for c in CONDS:
        bold = bold_set(cv, c)
        cells = []
        for a in ARMS:
            mean, std = run_spread(cv, c, a)
            if a == "frozen":
                cells.append(f"{mean:.1f} $\\pm$ {std:.1f}")
                continue
            gain, _ = paired_gain(cv, c, a)
            value = f"\\textbf{{{mean:.1f}}}" if a in bold else f"{mean:.1f}"
            cells.append(f"{value} $\\pm$ {std:.1f} ({gain:+.1f})")
        rows.append(f"{COND_LABEL[c]} & " + " & ".join(cells) + r" \\")
    return "\n".join(rows)


def per_seed_table(cv):
    rows = []
    for c in CONDS:
        for a in ARMS:
            v = cv[(c, a)][:, -1]
            per_seed = " & ".join(f"{x:.0f}" for x in v)
            rows.append(f"{COND_LABEL[c]} & {ARM_LABEL[a]} & {per_seed} & {v.mean():.1f} \\\\")
        rows.append(r"\midrule")
    return "\n".join(rows[:-1])


def step_report(cv, cond, step):
    for a in ARMS:
        if a == "frozen":
            continue
        d = cv[(cond, a)][:, step - 1] - cv[(cond, "frozen")][:, step - 1]
        print(f"  {a:10} {d.mean():+5.1f}   per-seed {np.round(d).astype(int).tolist()}")
    raw = cv[(cond, "alra")][:, step - 1]
    print(f"  alra raw rate  {raw.min():.0f}-{raw.max():.0f}%")


def main():
    cv = all_curves()

    bad = verify(cv)
    print(f"verification: {len(CONDS) * len(ARMS) * len(SEEDS)} runs, "
          + ("all match final_eval/success_rate" if not bad else f"MISMATCHES {bad}"))

    print("\n--- Table 1 ---")
    print(results_table(cv))

    print("\n--- Per-seed table ---")
    print(per_seed_table(cv))

    print("\n--- damping_x50 at step 12 ---")
    step_report(cv, "damping_x50", 12)


if __name__ == "__main__":
    main()
