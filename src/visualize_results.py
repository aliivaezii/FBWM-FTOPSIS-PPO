#!/usr/bin/env python3
"""
Unified Publication-Quality Visualization & Statistical Analysis Pipeline
=========================================================================

Generates Q1-journal-grade figures and rigorous statistical reports for
the FBWM-FTOPSIS + PPO multi-seed supply chain experiment.

Valid models (strictly):
    M1: Base-Stock   (heuristic baseline)
    M2: Vanilla PPO  (RL without MCDM priors)
    M3: PPO + Priors (RL with FTOPSIS-augmented state)

Figures produced:
    1  Multi-metric grouped bars  (cost, sustainability, bullwhip)
    2  Seed-level variability     (box + jitter strip)
    3  Pareto front               (cost vs sustainability scatter)
    4  Supplier allocation heatmap
    5  Bullwhip ratio comparison  (horizontal bars)
    6  Lambda sensitivity Pareto  (if data available)
    7  Learning curves            (mean +/- 1 std shaded band, from evaluations.npz)

Statistical outputs:
    statistical_report.txt    Welch t-tests + Cohen's d (M2 vs M3)
    anova_report.txt          One-way ANOVA per scenario

Usage:
    python src/visualize_results.py
    python src/visualize_results.py --results-dir X --output-dir Y
"""

from __future__ import annotations
import argparse, os, sys, warnings, pathlib, glob
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import matplotlib.colors as mcolors

from scipy import stats

# =====================================================================
# GLOBAL STYLE  (strict Q1 journal)
# =====================================================================
_SERIF = ["Times New Roman", "DejaVu Serif", "Georgia", "serif"]

def _apply_style():
    plt.rcParams.update({
        # -- fonts --
        "font.family":          "serif",
        "font.serif":           _SERIF,
        "font.size":            9,
        "axes.titlesize":       11,
        "axes.labelsize":       9.5,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "legend.fontsize":      8,
        "legend.title_fontsize": 8.5,
        # -- lines / markers --
        "lines.linewidth":      1.2,
        "lines.markersize":     4,
        # -- axes --
        "axes.linewidth":       0.6,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        # -- faint y-gridlines --
        "axes.grid":            True,
        "axes.grid.axis":       "y",
        "grid.color":           "#E0E0E0",
        "grid.linewidth":       0.4,
        "grid.linestyle":       "--",
        # -- ticks --
        "xtick.major.width":    0.5,
        "ytick.major.width":    0.5,
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        # -- figure / save --
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.08,
    })

_apply_style()

# =====================================================================
# COLOUR PALETTE  (high-contrast, colorblind-safe)
# =====================================================================
#   M1 = neutral grey   M2 = strong blue   M3 = warm orange
MODEL_COLOURS: Dict[str, str] = {
    "M1: Base-Stock":   "#888888",
    "M2: Vanilla PPO":  "#0077BB",
    "M3: PPO + Priors": "#EE7733",
}
MODEL_ORDER: List[str] = [
    "M1: Base-Stock",
    "M2: Vanilla PPO",
    "M3: PPO + Priors",
]
MODEL_SHORT: Dict[str, str] = {
    "M1: Base-Stock":   "M1 (Base-Stock)",
    "M2: Vanilla PPO":  "M2 (Vanilla PPO)",
    "M3: PPO + Priors": "M3 (PPO + Priors)",
}
SCENARIO_ORDER: List[str] = [
    "Stable_Operations",
    "High_Volatility",
    "Systemic_Shock",
]
SCENARIO_LABEL: Dict[str, str] = {
    "Stable_Operations": "Stable",
    "High_Volatility":   "High Volatility",
    "Systemic_Shock":    "Systemic Shock",
}
FTOPSIS_SCORES: Dict[str, float] = {
    "S1": 0.0451, "S2": 0.6911, "S3": 0.7962,
    "S4": 0.4941, "S5": 0.8808,
}

# Figure widths (inches) -- Elsevier single / double column
W1: float = 3.5
W2: float = 7.2   # slightly wider for breathing room

# Directory name mapping used in the experiment output tree
MODEL_DIR: Dict[str, str] = {
    "M2: Vanilla PPO":  "M2_Vanilla_PPO",
    "M3: PPO + Priors": "M3_PPO_+_Priors",
}

# =====================================================================
# I/O HELPERS
# =====================================================================
def _csv_paths(rd: pathlib.Path) -> Dict[str, pathlib.Path]:
    return {
        "full":  rd / "full_results.csv",
        "raw":   rd / "raw_episode_all.csv",
        "alloc": rd / "allocation_table.csv",
        "sens":  rd / "Sensitivity_Analysis" / "sensitivity_results.csv",
    }

def _load(path: pathlib.Path, required: bool = True) -> Optional[pd.DataFrame]:
    if path.exists():
        df = pd.read_csv(path)
        print(f"  loaded {path.name}  ({len(df)} rows)")
        return df
    if required:
        sys.exit(f"  MISSING required: {path}")
    print(f"  optional not found: {path.name}")
    return None

def _save(fig: plt.Figure, name: str, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    print(f"  -> {name}.pdf / .png")
    plt.close(fig)

# =====================================================================
# STATISTICS  (Welch t-tests, ANOVA, Cohen's d)
# =====================================================================
def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pool = np.sqrt(((na-1)*np.var(a,ddof=1)+(nb-1)*np.var(b,ddof=1))/(na+nb-2))
    return float((np.mean(a)-np.mean(b))/pool) if pool > 0 else 0.0

def _eff(d: float) -> str:
    d = abs(d)
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "large"

def run_statistical_report(raw: pd.DataFrame, out: pathlib.Path) -> str:
    """Welch t-test + Cohen's d for M2 vs M3 on three metrics, per scenario."""
    L: List[str] = []
    L.append("=" * 74)
    L.append("STATISTICAL REPORT  --  M2 vs M3  (Welch t-test, Cohen's d)")
    L.append("=" * 74)
    metrics = [
        ("total_cost",           "Total Cost (EUR)"),
        ("sustainability_score", "Sustainability Score"),
        ("bullwhip_ratio",       "Bullwhip Ratio"),
    ]
    for sc in SCENARIO_ORDER:
        L.append(f"\n--- Scenario: {SCENARIO_LABEL[sc]} ---")
        m2 = raw[(raw.scenario==sc)&(raw.model=="M2: Vanilla PPO")]
        m3 = raw[(raw.scenario==sc)&(raw.model=="M3: PPO + Priors")]
        for col, lab in metrics:
            a, b = m2[col].values, m3[col].values
            t, p = stats.ttest_ind(a, b, equal_var=False)
            d = _cohens_d(a, b)
            sig = "Yes" if p < 0.05 else "No"
            L.append(f"  {lab:<24s}  t={t:+8.3f}  p={p:.2e}  d={d:+.3f} ({_eff(d)})  sig={sig}")
            better = "M2" if np.mean(a) < np.mean(b) else "M3"
            if col == "sustainability_score":
                better = "M2" if np.mean(a) > np.mean(b) else "M3"
            L.append(f"    -> {better} favoured  (M2={np.mean(a):.4f}, M3={np.mean(b):.4f})")
    # cross-seed summary
    L.append(f"\n{'='*74}\nCROSS-SEED MEAN SUMMARY\n{'='*74}")
    s = raw.groupby(["scenario","model"]).agg(
        cost_m=("total_cost","mean"), cost_s=("total_cost","std"),
        sust_m=("sustainability_score","mean"), bull_m=("bullwhip_ratio","mean"),
    ).reset_index()
    for sc in SCENARIO_ORDER:
        L.append(f"\n  {SCENARIO_LABEL[sc]}:")
        for m in MODEL_ORDER:
            r = s[(s.scenario==sc)&(s.model==m)]
            if r.empty: continue
            r = r.iloc[0]
            L.append(f"    {MODEL_SHORT[m]}: cost=EUR {r.cost_m/1e6:.2f}M +/- {r.cost_s/1e6:.2f}M  "
                     f"sust={r.sust_m:.3f}  bull={r.bull_m:.2f}")
    txt = "\n".join(L)
    out.mkdir(parents=True, exist_ok=True)
    (out / "statistical_report.txt").write_text(txt)
    print("  -> statistical_report.txt")
    return txt


def run_anova_report(raw: pd.DataFrame, out: pathlib.Path) -> str:
    """One-way ANOVA across M1/M2/M3 per scenario."""
    L: List[str] = []
    L.append("=" * 74)
    L.append("ANOVA REPORT  --  M1 vs M2 vs M3  (one-way, per scenario)")
    L.append("=" * 74)
    for sc in SCENARIO_ORDER:
        L.append(f"\n--- {SCENARIO_LABEL[sc]} ---")
        groups = []
        for m in MODEL_ORDER:
            g = raw[(raw.scenario==sc)&(raw.model==m)]["total_cost"].values
            groups.append(g)
            L.append(f"    {MODEL_SHORT[m]:>25s}: n={len(g)}, mean=EUR {np.mean(g)/1e6:.2f}M")
        F, p = stats.f_oneway(*groups)
        L.append(f"    F={F:.3f}, p={p:.2e}  {'(significant)' if p<0.05 else '(not significant)'}")
    txt = "\n".join(L)
    out.mkdir(parents=True, exist_ok=True)
    (out / "anova_report.txt").write_text(txt)
    print("  -> anova_report.txt")
    return txt


# =====================================================================
# FIGURE 1  --  Multi-Metric Grouped Bar Chart
# =====================================================================
def figure1_performance_bars(full: pd.DataFrame, out: pathlib.Path) -> None:
    agg = full.groupby(["scenario","model"]).agg(
        cost_m=("total_cost_mean","mean"),  cost_e=("total_cost_mean","std"),
        sust_m=("sustainability_mean","mean"), sust_e=("sustainability_mean","std"),
        bull_m=("bullwhip_mean","mean"),  bull_e=("bullwhip_mean","std"),
    ).reset_index()

    panels = [
        ("cost_m","cost_e","Total Cost (M\u20ac)",1e6),
        ("sust_m","sust_e","Sustainability Score",1),
        ("bull_m","bull_e","Bullwhip Ratio",1),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.8))
    fig.subplots_adjust(top=0.82, bottom=0.18, wspace=0.38)
    x = np.arange(len(SCENARIO_ORDER))
    bw = 0.22
    off = np.array([(i - 1) * bw for i in range(3)])

    for pi, (mc, ec, yl, div) in enumerate(panels):
        ax = axes[pi]
        for mi, m in enumerate(MODEL_ORDER):
            v = []; e = []
            for sc in SCENARIO_ORDER:
                r = agg[(agg.scenario == sc) & (agg.model == m)]
                v.append(r[mc].values[0] / div if not r.empty else 0)
                ev = r[ec].values[0] / div if not r.empty else 0
                e.append(ev if not np.isnan(ev) else 0)
            # Asymmetric error bars: clamp lower bound so bars never go below 0
            v_arr = np.array(v)
            e_arr = np.array(e)
            e_lower = np.minimum(e_arr, v_arr)   # cannot exceed bar height
            e_upper = e_arr
            ax.bar(x + off[mi], v, bw, yerr=[e_lower, e_upper],
                   capsize=2, lw=0.5,
                   color=MODEL_COLOURS[m], edgecolor="white",
                   label=MODEL_SHORT[m] if pi == 0 else None, zorder=3,
                   error_kw=dict(lw=0.6, capthick=0.6))
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABEL[s] for s in SCENARIO_ORDER],
                           fontsize=7.5, rotation=20, ha="right")
        ax.set_ylabel(yl, fontsize=9)
        ax.set_title(f"({'abc'[pi]})", loc="left", fontweight="bold", fontsize=10.5)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.97), fontsize=8,
               columnspacing=1.8, handletextpad=0.5)
    _save(fig, "figure1_performance_comparison", out)


# =====================================================================
# FIGURE 2  --  Seed-Level Variability (box + strip)
# =====================================================================
def figure2_seed_variability(raw: pd.DataFrame, out: pathlib.Path) -> None:
    df = raw.copy()
    df["cost_M"] = df["total_cost"] / 1e6
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.7), sharey=False)
    fig.subplots_adjust(top=0.82, bottom=0.12, wspace=0.32)
    seed_mk = ["o", "s", "D"]
    seeds = sorted(df["seed"].unique())

    for i, sc in enumerate(SCENARIO_ORDER):
        ax = axes[i]
        sub = df[df.scenario == sc]
        for mi, m in enumerate(MODEL_ORDER):
            md = sub[sub.model == m]["cost_M"]
            ax.boxplot([md.values], positions=[mi], widths=0.48,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=MODEL_COLOURS[m], alpha=0.20, lw=0.6),
                       medianprops=dict(color="black", lw=1.0),
                       whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
            rng = np.random.default_rng(42)
            sa = sub[sub.model == m]["seed"].values
            for si, sv in enumerate(seeds):
                mask = sa == sv
                y = md.values[mask]
                jit = rng.uniform(-0.10, 0.10, len(y))
                ax.scatter(np.full_like(y, mi) + jit, y,
                           color=MODEL_COLOURS[m], edgecolors="white",
                           s=10, lw=0.3, marker=seed_mk[si % 3],
                           alpha=0.65, zorder=5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["M1", "M2", "M3"], fontsize=8)
        ax.set_title(f"({'abc'[i]}) {SCENARIO_LABEL[sc]}", loc="left",
                     fontweight="bold", fontsize=10)
        if i == 0:
            ax.set_ylabel("Total Cost (M\u20ac)", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))

    lh = [mlines.Line2D([], [], color="grey", marker=seed_mk[j], ls="None",
           ms=4, label=f"Seed {s}") for j, s in enumerate(seeds)]
    fig.legend(handles=lh, loc="upper center", ncol=len(seeds), frameon=False,
               bbox_to_anchor=(0.5, 0.97), fontsize=8,
               columnspacing=1.2, handletextpad=0.3)
    _save(fig, "figure2_seed_variability", out)


# =====================================================================
# FIGURE 3  --  Pareto Front (cost vs sustainability)
# =====================================================================
def figure3_pareto(raw: pd.DataFrame, out: pathlib.Path) -> None:
    sa = raw.groupby(["scenario", "model", "seed"]).agg(
        cost=("total_cost", "mean"),
        sust=("sustainability_score", "mean")).reset_index()
    sa["cost_M"] = sa["cost"] / 1e6
    mk = {"M1: Base-Stock": "^", "M2: Vanilla PPO": "o", "M3: PPO + Priors": "s"}

    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.7), sharey=True)
    fig.subplots_adjust(top=0.82, bottom=0.16, wspace=0.12)

    for i, sc in enumerate(SCENARIO_ORDER):
        ax = axes[i]
        sub = sa[sa.scenario == sc]
        for m in MODEL_ORDER:
            ms = sub[sub.model == m]
            # Individual seed runs (small, semi-transparent)
            ax.scatter(ms.cost_M, ms.sust, c=MODEL_COLOURS[m], marker=mk[m],
                       s=20, edgecolors="white", lw=0.3, alpha=0.6,
                       label=MODEL_SHORT[m] if i == 0 else None, zorder=5)
            # Cross-seed mean (larger, black edge)
            ax.scatter(ms.cost_M.mean(), ms.sust.mean(), c=MODEL_COLOURS[m],
                       marker=mk[m], s=80, edgecolors="black", lw=0.8, zorder=6)
        ax.set_xlabel("Total Cost (M\u20ac)", fontsize=9)
        if i == 0:
            ax.set_ylabel("Sustainability Score", fontsize=9)
        ax.set_title(f"({'abc'[i]}) {SCENARIO_LABEL[sc]}", loc="left",
                     fontweight="bold", fontsize=10)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.97), fontsize=8,
               columnspacing=1.5, handletextpad=0.4)
    _save(fig, "figure3_pareto_cost_sustainability", out)


# =====================================================================
# FIGURE 4  --  Supplier Allocation Heat-Map
# =====================================================================
def figure4_allocation_heatmap(alloc: pd.DataFrame, out: pathlib.Path) -> None:
    scols = ["S1_Orders", "S2_Orders", "S3_Orders", "S4_Orders", "S5_Orders"]
    slabs = ["S1", "S2", "S3", "S4", "S5"]
    df = alloc[alloc.Model.isin(["M2: Vanilla PPO", "M3: PPO + Priors"])].copy()
    for c in scols:
        df[c] = df[c] / df["Total_Orders"]
    rows_l = []; mat = []
    for sc in SCENARIO_ORDER:
        for m in ["M2: Vanilla PPO", "M3: PPO + Priors"]:
            r = df[(df.Scenario == sc) & (df.Model == m)]
            if r.empty:
                continue
            short = "M2" if "Vanilla" in m else "M3"
            rows_l.append(f"{SCENARIO_LABEL[sc]} \u2014 {short}")
            mat.append(r[scols].values[0])
    mat = np.array(mat)

    fig, ax = plt.subplots(figsize=(W1 + 1.5, W1 * 0.88), layout="constrained")
    cmap = plt.cm.YlOrRd
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=0.85)

    # Auto-contrast text: white on dark cells, black on light cells
    norm = mcolors.Normalize(vmin=0, vmax=0.85)
    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            v = mat[ri, ci]
            # Use luminance of the mapped colour to decide text colour
            rgba = cmap(norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            txt_col = "white" if lum < 0.55 else "black"
            ax.text(ci, ri, f"{v:.0%}", ha="center", va="center",
                    fontsize=8.5, color=txt_col, fontweight="bold")

    ax.set_xticks(range(5))
    ax.set_xticklabels(slabs, fontsize=9)
    ax.set_yticks(range(len(rows_l)))
    ax.set_yticklabels(rows_l, fontsize=8)

    # Secondary top axis for FTOPSIS scores — with extra padding
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(5))
    fv = [FTOPSIS_SCORES[s] for s in slabs]
    ax2.set_xticklabels([f"({v:.2f})" for v in fv], fontsize=7, color="#555")
    ax2.set_xlabel("(FTOPSIS Score)", fontsize=7.5, color="#555",
                   labelpad=4)
    ax2.tick_params(length=0, pad=3)

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cb.set_label("Order Share", fontsize=8.5)
    cb.ax.tick_params(labelsize=7)

    # Title with generous padding above the secondary axis
    ax.set_title("Supplier Order Allocation (RL Models)",
                 fontsize=11, pad=45, fontweight="bold")
    _save(fig, "figure4_allocation_heatmap", out)


# =====================================================================
# FIGURE 5  --  Bullwhip Ratio (horizontal bars)
# =====================================================================
def figure5_bullwhip(raw: pd.DataFrame, out: pathlib.Path) -> None:
    agg = raw.groupby(["scenario", "model"]).agg(
        bm=("bullwhip_ratio", "mean"),
        be=("bullwhip_ratio", lambda x: x.std() / np.sqrt(len(x)))).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.0), sharey=True, layout="constrained")

    for i, sc in enumerate(SCENARIO_ORDER):
        ax = axes[i]
        sub = agg[agg.scenario == sc]
        yp = np.arange(3)
        vs = []; es = []; cs = []
        for m in MODEL_ORDER:
            r = sub[sub.model == m]
            vs.append(r.bm.values[0] if not r.empty else 0)
            es.append(r.be.values[0] if not r.empty else 0)
            cs.append(MODEL_COLOURS[m])
        bars = ax.barh(yp, vs, xerr=es, height=0.52, color=cs,
                       edgecolor="white", capsize=2.5,
                       error_kw=dict(lw=0.7, capthick=0.6), zorder=3)
        # Place value text AFTER the error bar tip (not on top of it)
        for bi, v in enumerate(vs):
            if v > 0:
                err_tip = v + es[bi]  # end of the error bar
                pad = max(vs) * 0.05  # padding after the error bar cap
                ax.text(err_tip + pad, bi, f"{v:.1f}",
                        va="center", fontsize=7, color="#333")
        ax.set_yticks(yp)
        ax.set_yticklabels(["M1", "M2", "M3"], fontsize=8)
        ax.set_xlabel("Bullwhip Ratio", fontsize=9)
        ax.set_title(f"({'abc'[i]}) {SCENARIO_LABEL[sc]}", loc="left",
                     fontweight="bold", fontsize=10)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
        ax.axvline(1.0, color="#ccc", ls="--", lw=0.6, zorder=1)

    _save(fig, "figure5_bullwhip_comparison", out)


# =====================================================================
# FIGURE 6  --  Lambda Sensitivity Pareto (optional)
# =====================================================================
def figure6_sensitivity(sens: Optional[pd.DataFrame], out: pathlib.Path) -> None:
    if sens is None:
        print("  Skipping Figure 6 (no sensitivity data)")
        return
    agg = sens.groupby(["lambda_sust", "model"]).agg(
        cm=("total_cost_mean", "mean"), ce=("total_cost_mean", "std"),
        sm=("sustainability_mean", "mean"),
        se=("sustainability_mean", "std")).reset_index()
    agg["cM"] = agg.cm / 1e6
    agg["ceM"] = agg.ce / 1e6
    lams = sorted(agg.lambda_sust.unique())
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(lams), max(lams))

    fig, ax = plt.subplots(figsize=(W1 + 0.8, W1 * 0.95))
    fig.subplots_adjust(top=0.82, bottom=0.14)
    mk = {"M2: Vanilla PPO": "o", "M3: PPO + Priors": "s"}
    for m in ["M2: Vanilla PPO", "M3: PPO + Priors"]:
        sub = agg[agg.model == m]
        for _, r in sub.iterrows():
            c = cmap(norm(r.lambda_sust))
            ax.errorbar(r.cM, r.sm, xerr=r.ceM, yerr=r.se,
                        fmt=mk[m], color=c, ms=6, mec="white", mew=0.4,
                        elinewidth=0.7, capsize=2, zorder=5)
        ss = sub.sort_values("lambda_sust")
        ax.plot(ss.cM, ss.sm, ls="--", lw=0.8,
                color=MODEL_COLOURS[m], alpha=0.5, zorder=3)

    ax.set_xlabel("Total Cost (M\u20ac)", fontsize=9.5)
    ax.set_ylabel("Sustainability Score", fontsize=9.5)
    ax.set_title("$\\lambda_{sust}$ Sensitivity Analysis",
                 fontweight="bold", fontsize=11)
    sm_ = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm_.set_array([])
    cb = fig.colorbar(sm_, ax=ax, fraction=0.03, pad=0.04)
    cb.set_label("$\\lambda_{sust}$", fontsize=8.5)
    cb.ax.tick_params(labelsize=7)

    lh = [mlines.Line2D([], [], color=MODEL_COLOURS["M2: Vanilla PPO"],
            marker="o", ls="--", ms=5, label="M2 (Vanilla PPO)"),
          mlines.Line2D([], [], color=MODEL_COLOURS["M3: PPO + Priors"],
            marker="s", ls="--", ms=5, label="M3 (PPO + Priors)")]
    ax.legend(handles=lh, loc="upper center", ncol=2, frameon=False,
              bbox_to_anchor=(0.5, 1.05), fontsize=8)
    _save(fig, "figure6_sensitivity_pareto", out)


# =====================================================================
# FIGURE 7  --  Learning Curves (from evaluations.npz, multi-seed)
# =====================================================================
def _load_eval_npz(results_dir: pathlib.Path, scenario: str,
                   model_dir: str, seeds: List[int]
                   ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load evaluations.npz across seeds.  Returns (timesteps, reward_matrix)
    where reward_matrix has shape (n_seeds, n_checkpoints).
    """
    all_rewards = []
    common_ts = None
    for seed in seeds:
        p = results_dir / scenario / model_dir / f"seed_{seed}" / "evaluations.npz"
        if not p.exists():
            continue
        data = np.load(str(p))
        ts = data["timesteps"]
        rw = data["results"]
        if rw.ndim > 1:
            rw = rw.mean(axis=1)  # mean across eval episodes
        if common_ts is None:
            common_ts = ts
        # align lengths (different seeds may have slight differences)
        minlen = min(len(common_ts), len(rw))
        common_ts = common_ts[:minlen]
        rw = rw[:minlen]
        all_rewards.append(rw)
    if not all_rewards:
        return None
    # trim all to same length
    minl = min(len(r) for r in all_rewards)
    common_ts = common_ts[:minl]
    mat = np.array([r[:minl] for r in all_rewards])
    return common_ts, mat


def figure7_learning_curves(results_dir: pathlib.Path, out: pathlib.Path) -> None:
    """
    Learning curves: 1x3 subplots (one per scenario).
    Each panel shows M2 and M3 with mean +/- 1 std shaded.
    """
    # Discover seeds from directory structure
    seed_dirs = sorted(glob.glob(
        str(results_dir / "Stable_Operations" / "M2_Vanilla_PPO" / "seed_*")))
    if not seed_dirs:
        print("  Skipping Figure 7 (no per-seed evaluation data found)")
        return
    seeds = [int(os.path.basename(d).replace("seed_", "")) for d in seed_dirs]
    print(f"  Seeds detected: {seeds}")

    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.7), sharey=False)
    fig.subplots_adjust(top=0.82, bottom=0.18, wspace=0.32)

    rl_models = [("M2: Vanilla PPO", "M2_Vanilla_PPO"),
                 ("M3: PPO + Priors", "M3_PPO_+_Priors")]

    for i, sc in enumerate(SCENARIO_ORDER):
        ax = axes[i]
        for m_name, m_dir in rl_models:
            loaded = _load_eval_npz(results_dir, sc, m_dir, seeds)
            if loaded is None:
                continue
            ts, mat = loaded
            ts_k = ts / 1e3  # convert to thousands
            mean = mat.mean(axis=0)
            std = mat.std(axis=0)
            colour = MODEL_COLOURS[m_name]
            short = "M2" if "Vanilla" in m_name else "M3"
            ax.plot(ts_k, mean, color=colour, lw=1.3,
                    label=short if i == 0 else None, zorder=4)
            ax.fill_between(ts_k, mean - std, mean + std,
                            color=colour, alpha=0.15, zorder=2)

        ax.set_xlabel("Training Steps ($\\times 10^3$)", fontsize=9)
        if i == 0:
            ax.set_ylabel("Mean Eval. Reward", fontsize=9)
        ax.set_title(f"({'abc'[i]}) {SCENARIO_LABEL[sc]}",
                     loc="left", fontweight="bold", fontsize=10)

        # Clean x-axis: integer thousands, no decimals
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5, integer=True))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{int(v)}" if v == int(v) else f"{v:.0f}"))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.97), fontsize=8.5,
               columnspacing=2.0, handletextpad=0.5)
    _save(fig, "figure7_learning_curves", out)


# =====================================================================
# MAIN
# =====================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Unified visualisation pipeline")
    ap.add_argument("--results-dir", default="results/experiments")
    ap.add_argument("--output-dir",  default="results/figures")
    args = ap.parse_args()

    proj = pathlib.Path(__file__).resolve().parent.parent
    rd  = (proj / args.results_dir).resolve()
    out = (proj / args.output_dir).resolve()

    print(f"\n{'='*62}")
    print(f"  Unified Visualisation & Statistics Pipeline")
    print(f"  Results : {rd}")
    print(f"  Output  : {out}")
    print(f"{'='*62}\n")

    paths = _csv_paths(rd)
    full  = _load(paths["full"])
    raw   = _load(paths["raw"])
    alloc = _load(paths["alloc"])
    sens  = _load(paths["sens"], required=False)

    print("\n-- Statistical Reports --")
    run_statistical_report(raw, out)
    run_anova_report(raw, out)

    print("\n-- Figure 1: Performance Comparison --")
    figure1_performance_bars(full, out)

    print("\n-- Figure 2: Seed Variability --")
    figure2_seed_variability(raw, out)

    print("\n-- Figure 3: Pareto Front --")
    figure3_pareto(raw, out)

    print("\n-- Figure 4: Allocation Heatmap --")
    figure4_allocation_heatmap(alloc, out)

    print("\n-- Figure 5: Bullwhip Comparison --")
    figure5_bullwhip(raw, out)

    print("\n-- Figure 6: Sensitivity (if available) --")
    figure6_sensitivity(sens, out)

    print("\n-- Figure 7: Learning Curves --")
    figure7_learning_curves(rd, out)

    print(f"\n{'='*62}")
    print(f"  Done. All outputs in {out}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
