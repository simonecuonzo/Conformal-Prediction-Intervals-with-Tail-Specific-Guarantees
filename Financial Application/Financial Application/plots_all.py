# ============================================================
# STYLED COMBINED CQR + RES PLOTS
#
# Produces:
#   1) Prediction intervals
#   2) Lower coverage over time
#   3) Upper coverage over time
#
# Assets:
#   - TQQQ
#   - XLE
#   - SPY
#
# Style logic:
#   - color identifies the score type: q, res, s-res
#   - linestyle identifies the method type:
#       one-sided  -> solid
#       two-sided  -> dashed
#       benchmark  -> dotted
#   - target coverage line is gray and thin
# ============================================================

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# STYLE OPTIONS
# ============================================================

# Choose:
#   "all"      -> title + legend in all figures
#   "spy_only" -> title + legend only for SPY
#   "none"     -> no title and no legend in all figures
TITLE_LEGEND_MODE = "spy_only"

TITLE_SIZE = 38
X_LABEL_SIZE = 24
TICK_SIZE = 19
LEGEND_SIZE = 17

LW_MAIN = 2.3
LW_TARGET = 0.8
LW_RETURNS = 0.7


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

output_dir = "./styled_combined_plots"
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

with open("equity_results_garch_CQR_FACI.pkl", "rb") as f:
    cqr_equity = pickle.load(f)["results_all"]

with open("equity_results_garch_FACI.pkl", "rb") as f:
    res_equity = pickle.load(f)["results_all"]

with open("spy_results_garch_CQR_FACI.pkl", "rb") as f:
    cqr_spy = pickle.load(f)["results_all"]

with open("spy_results_garch_FACI.pkl", "rb") as f:
    res_spy = pickle.load(f)["results_all"]


# ============================================================
# MERGE CQR RESULTS
# ============================================================

results_cqr = {}

for model in cqr_equity:
    results_cqr[model] = {}
    results_cqr[model].update(cqr_equity[model])

    if model in cqr_spy:
        results_cqr[model].update(cqr_spy[model])


# ============================================================
# MERGE RES RESULTS
# ============================================================

results_res = {}

for model in res_equity:
    results_res[model] = {}
    results_res[model].update(res_equity[model])

    if model in res_spy:
        results_res[model].update(res_spy[model])


# ============================================================
# ASSETS
# ============================================================

assets = ["TQQQ", "XLE", "SPY"]


# ============================================================
# MODELS
# ============================================================

all_models = [
    "q (One-sided)",
    "q (Two-sided)",
    "res (One-sided)",
    "s-res (One-sided)",
    "res (Two-sided)",
    "s-res (Two-sided)",
]


# ============================================================
# COLORS: same score = same color
# ============================================================

score_colors = {
    "q": "#1f77b4",       # blue
    "res": "#2ca02c",     # green
    "s-res": "#d62728",   # red
    "raw": "black",
    "target": "gray",
}


# ============================================================
# LINESTYLES: same method = same linestyle
# ============================================================

method_linestyles = {
    "one-sided": "-",
    "two-sided": "--",
    "benchmark": ":",
    "target": "--",
}


# ============================================================
# MODEL -> SCORE TYPE
# ============================================================

model_to_score = {
    "q (One-sided)": "q",
    "q (Two-sided)": "q",

    "res (One-sided)": "res",
    "res (Two-sided)": "res",

    "s-res (One-sided)": "s-res",
    "s-res (Two-sided)": "s-res",
}


# ============================================================
# MODEL -> METHOD TYPE
# ============================================================

model_to_method = {
    "q (One-sided)": "one-sided",
    "q (Two-sided)": "two-sided",

    "res (One-sided)": "one-sided",
    "res (Two-sided)": "two-sided",

    "s-res (One-sided)": "one-sided",
    "s-res (Two-sided)": "two-sided",
}


# ============================================================
# STYLE HELPERS
# ============================================================

def get_model_color(model):
    return score_colors[model_to_score[model]]


def get_model_linestyle(model):
    return method_linestyles[model_to_method[model]]


def use_full_style(asset):
    if TITLE_LEGEND_MODE == "all":
        return True

    if TITLE_LEGEND_MODE == "spy_only":
        return asset == "SPY"

    if TITLE_LEGEND_MODE == "none":
        return False

    raise ValueError(
        "TITLE_LEGEND_MODE must be 'all', 'spy_only', or 'none'"
    )


# ============================================================
# CUSTOM FACTORIAL LEGEND
# ============================================================

def add_custom_legend(ax):
    """
    Legend with 6 elements:

    1) color legend:
        res
        s-res
        q

    2) linestyle legend:
        one-sided
        two-sided
        benchmark
    """

    handles = [
        # ----------------------------------------------------
        # Colors = score type
        # ----------------------------------------------------
        Line2D(
            [0],
            [0],
            color=score_colors["res"],
            linestyle="-",
            linewidth=LW_MAIN,
            label=r"$\mathrm{res}$"
        ),

        Line2D(
            [0],
            [0],
            color=score_colors["s-res"],
            linestyle="-",
            linewidth=LW_MAIN,
            label=r"$\mathrm{s\!-\!res}$"
        ),

        Line2D(
            [0],
            [0],
            color=score_colors["q"],
            linestyle="-",
            linewidth=LW_MAIN,
            label=r"$q$"
        ),

        # ----------------------------------------------------
        # Linestyles = method type
        # ----------------------------------------------------
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=method_linestyles["one-sided"],
            linewidth=LW_MAIN,
            label=r"$\mathcal{C}^{\cap}$"
        ),

        Line2D(
            [0],
            [0],
            color="black",
            linestyle=method_linestyles["two-sided"],
            linewidth=LW_MAIN,
            label=r"$\mathcal{C}$"
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=method_linestyles["benchmark"],
            linewidth=LW_MAIN,
            label=r"$\mathrm{Benchmark}$"
        ),
    ]

    ax.legend(
        handles=handles,
        fontsize=LEGEND_SIZE,
        ncol=2,
        frameon=True
    )


# ============================================================
# DATA HELPER
# ============================================================

def get_model_df(model, asset):
    """
    Returns the DataFrame for a given model and asset.
    Returns None if unavailable.
    """

    if model.startswith("q"):
        if model not in results_cqr:
            return None
        if asset not in results_cqr[model]:
            return None
        return results_cqr[model][asset]

    if model not in results_res:
        return None
    if asset not in results_res[model]:
        return None

    return results_res[model][asset]


# ============================================================
# COVERAGE FUNCTIONS
# ============================================================

def cumulative_lower_coverage(y, q):
    y = np.asarray(y, float)
    q = np.asarray(q, float)

    covered = (y >= q).astype(int)

    return np.cumsum(covered) / np.arange(1, len(covered) + 1)


def cumulative_upper_coverage(y, q):
    y = np.asarray(y, float)
    q = np.asarray(q, float)

    covered = (y <= q).astype(int)

    return np.cumsum(covered) / np.arange(1, len(covered) + 1)


# ============================================================
# TARGET COVERAGE
# ============================================================

alpha_target = 0.10
target_cov = 1.0 - alpha_target


# ============================================================
# DEBUG PRINT
# ============================================================

print("\n================ AVAILABLE DATA ================")

for model in all_models:
    if model.startswith("q"):
        available = list(results_cqr.get(model, {}).keys())
    else:
        available = list(results_res.get(model, {}).keys())

    print(f"{model}: {available}")

print("================================================\n")


# ============================================================
# MAIN LOOP
# ============================================================

saved_files = []

for asset in assets:

    print(f"\nProcessing asset: {asset}")

    baseline_df = get_model_df("q (One-sided)", asset)

    if baseline_df is None:
        print(f"⚠️ Skipping {asset}: missing q (One-sided) baseline")
        continue

    dates = baseline_df.index
    y = baseline_df["r"].values


    # ========================================================
    # 1) PREDICTION INTERVALS
    # ========================================================

    fig, ax = plt.subplots(figsize=(16, 7))

    # --------------------------------------------------------
    # returns
    # --------------------------------------------------------
    ax.plot(
        dates,
        y,
        color="black",
        linewidth=LW_RETURNS
    )

    # --------------------------------------------------------
    # raw benchmark lower / upper
    # same color and same linestyle
    # only one legend element is handled by custom legend
    # --------------------------------------------------------
    ax.plot(
        dates,
        baseline_df["q_lower_raw"],
        color=score_colors["raw"],
        linestyle=method_linestyles["benchmark"],
        linewidth=LW_MAIN
    )

    ax.plot(
        dates,
        baseline_df["q_upper_raw"],
        color=score_colors["raw"],
        linestyle=method_linestyles["benchmark"],
        linewidth=LW_MAIN
    )

    # --------------------------------------------------------
    # CP methods lower / upper
    # same method: same color and same linestyle
    # --------------------------------------------------------
    for model in all_models:

        df = get_model_df(model, asset)

        if df is None:
            print(f"⚠️ Missing {model} for {asset}; skipped in PI plot")
            continue

        ax.plot(
            df.index,
            df["L"],
            color=get_model_color(model),
            linestyle=get_model_linestyle(model),
            linewidth=LW_MAIN
        )

        ax.plot(
            df.index,
            df["U"],
            color=get_model_color(model),
            linestyle=get_model_linestyle(model),
            linewidth=LW_MAIN
        )

    # --------------------------------------------------------
    # style
    # --------------------------------------------------------
    ax.set_xlabel("Date", fontsize=X_LABEL_SIZE)
    ax.set_ylabel("")

    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    ax.grid(True, alpha=0.25)

    if use_full_style(asset):
        ax.set_title("Prediction intervals", fontsize=TITLE_SIZE)
        add_custom_legend(ax)
    else:
        ax.set_title("")

    fig.tight_layout()

    out_path = os.path.join(
        output_dir,
        f"PREDICTION_INTERVALS_{asset}.png"
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    saved_files.append(out_path)
    print(f"✅ Saved: {out_path}")


    # ========================================================
    # 2) LOWER COVERAGE
    # ========================================================

    fig, ax = plt.subplots(figsize=(14, 7))

    # --------------------------------------------------------
    # raw benchmark
    # --------------------------------------------------------
    cov_raw = cumulative_lower_coverage(
        y,
        baseline_df["q_lower_raw"].values
    )

    ax.plot(
        dates,
        cov_raw,
        color=score_colors["raw"],
        linestyle=method_linestyles["benchmark"],
        linewidth=LW_MAIN
    )

    # --------------------------------------------------------
    # CP methods
    # --------------------------------------------------------
    for model in all_models:

        df = get_model_df(model, asset)

        if df is None:
            print(f"⚠️ Missing {model} for {asset}; skipped in lower coverage")
            continue

        cov = cumulative_lower_coverage(
            df["r"].values,
            df["L"].values
        )

        ax.plot(
            df.index,
            cov,
            color=get_model_color(model),
            linestyle=get_model_linestyle(model),
            linewidth=LW_MAIN
        )

    # --------------------------------------------------------
    # target coverage line: gray and thin
    # --------------------------------------------------------
    ax.axhline(
        target_cov,
        color=score_colors["target"],
        linestyle=method_linestyles["target"],
        linewidth=LW_TARGET
    )

    # --------------------------------------------------------
    # style
    # --------------------------------------------------------
    ax.set_xlabel("Date", fontsize=X_LABEL_SIZE)
    ax.set_ylabel("")

    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    ax.set_ylim(0.8, 1.0)

    ax.grid(True, alpha=0.25)

    if use_full_style(asset):
        ax.set_title("Lower coverage over time", fontsize=TITLE_SIZE)
        add_custom_legend(ax)
    else:
        ax.set_title("")

    fig.tight_layout()

    out_path = os.path.join(
        output_dir,
        f"LOWER_COVERAGE_{asset}.png"
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    saved_files.append(out_path)
    print(f"✅ Saved: {out_path}")


    # ========================================================
    # 3) UPPER COVERAGE
    # ========================================================

    fig, ax = plt.subplots(figsize=(14, 7))

    # --------------------------------------------------------
    # raw benchmark
    # --------------------------------------------------------
    cov_raw = cumulative_upper_coverage(
        y,
        baseline_df["q_upper_raw"].values
    )

    ax.plot(
        dates,
        cov_raw,
        color=score_colors["raw"],
        linestyle=method_linestyles["benchmark"],
        linewidth=LW_MAIN
    )

    # --------------------------------------------------------
    # CP methods
    # --------------------------------------------------------
    for model in all_models:

        df = get_model_df(model, asset)

        if df is None:
            print(f"⚠️ Missing {model} for {asset}; skipped in upper coverage")
            continue

        cov = cumulative_upper_coverage(
            df["r"].values,
            df["U"].values
        )

        ax.plot(
            df.index,
            cov,
            color=get_model_color(model),
            linestyle=get_model_linestyle(model),
            linewidth=LW_MAIN
        )

    # --------------------------------------------------------
    # target coverage line: gray and thin
    # --------------------------------------------------------
    ax.axhline(
        target_cov,
        color=score_colors["target"],
        linestyle=method_linestyles["target"],
        linewidth=LW_TARGET
    )

    # --------------------------------------------------------
    # style
    # --------------------------------------------------------
    ax.set_xlabel("Date", fontsize=X_LABEL_SIZE)
    ax.set_ylabel("")

    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    ax.set_ylim(0.8, 1.0)

    ax.grid(True, alpha=0.25)

    if use_full_style(asset):
        ax.set_title("Upper coverage over time", fontsize=TITLE_SIZE)
        add_custom_legend(ax)
    else:
        ax.set_title("")

    fig.tight_layout()

    out_path = os.path.join(
        output_dir,
        f"UPPER_COVERAGE_{asset}.png"
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    saved_files.append(out_path)
    print(f"✅ Saved: {out_path}")


print("\n✅ DONE")
print(f"📊 Total figures saved: {len(saved_files)}")

for path in saved_files:
    print(path)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# ============================================================
# MULTI-PANEL FIGURES
#
# 1) 3x2 figure:
#       left  column = lower coverage
#       right column = prediction intervals
#
# 2) 1x3 figure:
#       upper coverage for SPY, TQQQ, XLE
#
# ============================================================

from matplotlib.lines import Line2D


# ============================================================
# PANEL STYLE
# ============================================================

PANEL_TITLE_SIZE = 20
PANEL_LABEL_SIZE = 18
PANEL_TICK_SIZE = 8
PANEL_LEGEND_SIZE = 7

LW_PANEL = 1.0
LW_PANEL_TARGET = 0.6
LW_PANEL_RETURNS = 0.45

PANEL_ASSETS = ["SPY", "TQQQ", "XLE"]


# ============================================================
# CUSTOM LEGEND FOR MULTI-PANEL FIGURES
# ============================================================

def add_small_custom_legend(ax):
    """
    Factorized legend:
        colors     = score type
        linestyles = method type
    """

    handles = [
        # score colors
        Line2D(
            [0], [0],
            color=score_colors["res"],
            linestyle="-",
            linewidth=LW_PANEL,
            label=r"$\mathrm{res}$"
        ),
        Line2D(
            [0], [0],
            color=score_colors["s-res"],
            linestyle="-",
            linewidth=LW_PANEL,
            label=r"$\mathrm{s\!-\!res}$"
        ),
        Line2D(
            [0], [0],
            color=score_colors["q"],
            linestyle="-",
            linewidth=LW_PANEL,
            label=r"$q$"
        ),

        # method linestyles
        Line2D(
            [0], [0],
            color="black",
            linestyle=method_linestyles["one-sided"],
            linewidth=LW_PANEL,
            label=r"$\mathcal{C}^{\cap}$"
        ),
        Line2D(
            [0], [0],
            color="black",
            linestyle=method_linestyles["two-sided"],
            linewidth=LW_PANEL,
            label=r"$\mathcal{C}$"
        ),
        Line2D(
            [0], [0],
            color="black",
            linestyle=method_linestyles["benchmark"],
            linewidth=LW_PANEL,
            label=r"$\mathrm{Benchmark}$"
        ),
    ]

    ax.legend(
        handles=handles,
        fontsize=PANEL_LEGEND_SIZE,
        ncol=2,
        frameon=True,
        loc="upper right"
    )


# ============================================================
# HELPER: PLOT LOWER COVERAGE ON AXIS
# ============================================================

def plot_lower_coverage_panel(ax, asset, show_legend=False, show_title=False):

    baseline_df = get_model_df("q (One-sided)", asset)

    if baseline_df is None:
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"Missing {asset}", ha="center", va="center")
        return

    dates = baseline_df.index
    y = baseline_df["r"].values

    # benchmark
    cov_raw = cumulative_lower_coverage(
        y,
        baseline_df["q_lower_raw"].values
    )

    ax.plot(
        dates,
        cov_raw,
        color=score_colors["raw"],
        linestyle=method_linestyles["benchmark"],
        linewidth=LW_PANEL
    )

    # CP models
    for model in all_models:

        df = get_model_df(model, asset)

        if df is None:
            continue

        cov = cumulative_lower_coverage(
            df["r"].values,
            df["L"].values
        )

        ax.plot(
            df.index,
            cov,
            color=get_model_color(model),
            linestyle=get_model_linestyle(model),
            linewidth=LW_PANEL
        )

    # target line
    ax.axhline(
        target_cov,
        color=score_colors["target"],
        linestyle=method_linestyles["target"],
        linewidth=LW_PANEL_TARGET
    )

    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("")
    ax.set_xlabel("Date", fontsize=PANEL_TICK_SIZE)
    ax.tick_params(axis="both", labelsize=PANEL_TICK_SIZE)
    ax.grid(True, alpha=0.18)

    if show_title:
        ax.set_title("Lower coverage over time", fontsize=PANEL_TITLE_SIZE)

    if show_legend:
        add_small_custom_legend(ax)


# ============================================================
# HELPER: PLOT UPPER COVERAGE ON AXIS
# ============================================================

def plot_upper_coverage_panel(ax, asset, show_legend=False, show_title=False):

    baseline_df = get_model_df("q (One-sided)", asset)

    if baseline_df is None:
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"Missing {asset}", ha="center", va="center")
        return

    dates = baseline_df.index
    y = baseline_df["r"].values

    # benchmark
    cov_raw = cumulative_upper_coverage(
        y,
        baseline_df["q_upper_raw"].values
    )

    ax.plot(
        dates,
        cov_raw,
        color=score_colors["raw"],
        linestyle=method_linestyles["benchmark"],
        linewidth=LW_PANEL
    )

    # CP models
    for model in all_models:

        df = get_model_df(model, asset)

        if df is None:
            continue

        cov = cumulative_upper_coverage(
            df["r"].values,
            df["U"].values
        )

        ax.plot(
            df.index,
            cov,
            color=get_model_color(model),
            linestyle=get_model_linestyle(model),
            linewidth=LW_PANEL
        )

    # target line
    ax.axhline(
        target_cov,
        color=score_colors["target"],
        linestyle=method_linestyles["target"],
        linewidth=LW_PANEL_TARGET
    )

    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("")
    ax.set_xlabel("Date", fontsize=PANEL_TICK_SIZE)
    ax.tick_params(axis="both", labelsize=PANEL_TICK_SIZE)
    ax.grid(True, alpha=0.18)

    if show_title:
        ax.set_title("Upper coverage over time", fontsize=PANEL_TITLE_SIZE)

    if show_legend:
        add_small_custom_legend(ax)


# ============================================================
# HELPER: PLOT PREDICTION INTERVALS ON AXIS
# ============================================================

def plot_prediction_intervals_panel(ax, asset, show_legend=False, show_title=False):

    baseline_df = get_model_df("q (One-sided)", asset)

    if baseline_df is None:
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"Missing {asset}", ha="center", va="center")
        return

    dates = baseline_df.index
    y = baseline_df["r"].values

    # returns
    ax.plot(
        dates,
        y,
        color="black",
        linewidth=LW_PANEL_RETURNS,
        alpha=0.75
    )

    # benchmark lower / upper
    ax.plot(
        dates,
        baseline_df["q_lower_raw"],
        color=score_colors["raw"],
        linestyle=method_linestyles["benchmark"],
        linewidth=LW_PANEL
    )

    ax.plot(
        dates,
        baseline_df["q_upper_raw"],
        color=score_colors["raw"],
        linestyle=method_linestyles["benchmark"],
        linewidth=LW_PANEL
    )

    # CP models lower / upper
    for model in all_models:

        df = get_model_df(model, asset)

        if df is None:
            continue

        ax.plot(
            df.index,
            df["L"],
            color=get_model_color(model),
            linestyle=get_model_linestyle(model),
            linewidth=LW_PANEL
        )

        ax.plot(
            df.index,
            df["U"],
            color=get_model_color(model),
            linestyle=get_model_linestyle(model),
            linewidth=LW_PANEL
        )

    ax.set_ylabel("")
    ax.set_xlabel("Date", fontsize=PANEL_TICK_SIZE)
    ax.tick_params(axis="both", labelsize=PANEL_TICK_SIZE)
    ax.grid(True, alpha=0.18)

    if show_title:
        ax.set_title("Prediction intervals", fontsize=PANEL_TITLE_SIZE)

    if show_legend:
        add_small_custom_legend(ax)


# ============================================================
# 1) FIGURE 3x2:
#    LOWER COVERAGE + PREDICTION INTERVALS
# ============================================================

fig, axes = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(12, 10),
    constrained_layout=False
)

panel_labels_3x2 = [
    "(a) SPY",
    "(b) SPY",
    "(c) TQQQ",
    "(d) TQQQ",
    "(e) XLE",
    "(f) XLE",
]

for i, asset in enumerate(PANEL_ASSETS):

    # left column: lower coverage
    plot_lower_coverage_panel(
        axes[i, 0],
        asset,
        show_legend=(i == 0),
        show_title=(i == 0)
    )

    # right column: prediction intervals
    plot_prediction_intervals_panel(
        axes[i, 1],
        asset,
        show_legend=False,
        show_title=(i == 0)
    )

# panel captions under each axis
for ax, lab in zip(axes.flatten(), panel_labels_3x2):

    ax.text(
        0.5,
        -0.35,
        lab,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PANEL_LABEL_SIZE
    )

fig.subplots_adjust(
    left=0.07,
    right=0.98,
    top=0.95,
    bottom=0.08,
    wspace=0.16,
    hspace=0.75
)

out_path = os.path.join(
    output_dir,
    "MULTIPANEL_LOWER_COVERAGE_AND_PI_3x2.png"
)

fig.savefig(
    out_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(f"✅ Saved multipanel 3x2 figure: {out_path}")


# ============================================================
# 2) FIGURE 3x1:
#    UPPER COVERAGE
# ============================================================

fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(7.5, 10.5),
    constrained_layout=False
)

panel_labels_upper = [
    "(a) SPY",
    "(b) TQQQ",
    "(c) XLE",
]

for j, asset in enumerate(PANEL_ASSETS):

    plot_upper_coverage_panel(
        axes[j],
        asset,
        show_legend=(j == 0),
        show_title=(j == 0)
    )

    axes[j].text(
        0.5,
        -0.32,
        panel_labels_upper[j],
        transform=axes[j].transAxes,
        ha="center",
        va="top",
        fontsize=PANEL_LABEL_SIZE
    )

fig.subplots_adjust(
    left=0.10,
    right=0.98,
    top=0.95,
    bottom=0.06,
    hspace=0.70
)

out_path = os.path.join(
    output_dir,
    "MULTIPANEL_UPPER_COVERAGE_3x1.png"
)

fig.savefig(
    out_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(f"✅ Saved upper coverage multipanel figure: {out_path}")