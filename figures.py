"""Draw the five figures used in the revision. All test traces use seed 101."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CN = [
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK JP",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": CN,
        "font.size": 9,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.labelcolor": "#1b293b",
        "text.color": "#16263c",
        "xtick.color": "#344455",
        "ytick.color": "#344455",
        "savefig.dpi": 400,
    }
)
BLUE = "#225498"
ORANGE = "#b04e16"
GREEN = "#18765b"
GRAY = "#4d5564"
LIGHT = "#dce7f6"
GRIDL = "#dce3eb"


def save(fig, loc, name):
    fig.savefig(
        loc / (name + ".png"), bbox_inches="tight", pad_inches=0.04, facecolor="white"
    )
    plt.close(fig)


def diagram(loc):
    fig, ax = plt.subplots(figsize=(7.15, 2.42))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.7)
    ax.axis("off")

    def box(x, y, w, h, s, fill="#f0f5fb"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=.035,rounding_size=.065",
                linewidth=0.85,
                edgecolor=BLUE,
                facecolor=fill,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            s,
            ha="center",
            va="center",
            color=BLUE,
            fontsize=9.3,
            linespacing=1.3,
        )

    def arrow(p, q):
        ax.add_patch(
            FancyArrowPatch(
                p,
                q,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.95,
                color=GRAY,
                shrinkA=1,
                shrinkB=1,
            )
        )

    box(0.05, 2.52, 2.72, 0.78, "已失效设备\n监测窗口及真实 RUL")
    box(3.58, 2.52, 2.77, 0.78, "自举训练与袋外预测\n估计中心和局部尺度")
    box(7.18, 2.52, 2.72, 0.78, "校准队列\n缩放残差的经验分位数")
    arrow((2.81, 2.91), (3.54, 2.91))
    arrow((6.39, 2.91), (7.14, 2.91))
    box(0.05, 0.92, 2.72, 0.77, "在役设备\n截至当前的监测窗口")
    box(
        3.58, 0.92, 2.77, 0.77, "计算中心和区间半宽\n发布并保存本次预测", fill="#e4edf8"
    )
    arrow((2.81, 1.31), (3.54, 1.31))
    arrow((4.96, 2.48), (4.96, 1.73))
    arrow((8.50, 2.48), (6.39, 1.54))
    box(7.18, 0.22, 2.72, 0.78, "确认失效或标签到达\n按到达次序补入残差")
    arrow((6.39, 1.09), (7.14, 0.77))
    arrow((9.43, 1.03), (9.43, 2.47))
    ax.text(3.62, 0.49, "无标签时沿用现有分位数", color=BLUE, fontsize=8.6)
    save(fig, loc, "fig1")


def architecture(loc):
    fig, ax = plt.subplots(figsize=(7.15, 1.76))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    a = [
        (0.02, 1.49, "监测窗口", "50 × 17"),
        (2.57, 1.49, "卷积 18 / 池化", "25 × 18"),
        (5.12, 1.49, "卷积 36 / 池化", "13 × 36"),
        (7.67, 1.49, "卷积 72 / 池化", "7 × 72"),
        (7.67, 0.18, "重排", "6 × 84"),
        (5.12, 0.18, "LSTM 1", "6 × 51"),
        (2.57, 0.18, "LSTM 2", "51"),
        (0.02, 0.18, "全连接 / 输出", "50 → 1"),
    ]
    for x, y, h, s in a:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                2.18,
                0.78,
                boxstyle="round,pad=.025,rounding_size=.055",
                edgecolor=BLUE,
                facecolor="#f0f5fb",
                linewidth=0.85,
            )
        )
        ax.text(
            x + 1.09, y + 0.55, h, ha="center", va="center", color=BLUE, fontsize=9.8
        )
        ax.text(x + 1.09, y + 0.23, s, ha="center", va="center", color=BLUE, fontsize=9)
    for x in [0.02, 2.57, 5.12]:
        ax.annotate(
            "",
            xy=(x + 2.53, 1.89),
            xytext=(x + 2.19, 1.89),
            arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": GRAY},
        )
        ax.annotate(
            "",
            xy=(x + 2.2, 0.56),
            xytext=(x + 2.53, 0.56),
            arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": GRAY},
        )
    ax.annotate(
        "",
        xy=(8.76, 0.98),
        xytext=(8.76, 1.46),
        arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": GRAY},
    )
    save(fig, loc, "fig3")


def simulation(loc):
    t = np.linspace(0, 1, 200)
    u = 100 * np.sin(2 * np.pi * t)
    std = [
        1 + 5 * t,
        np.abs(1 + 5 * np.sin(2 * np.pi * t)),
        np.where(t < 0.5, 1.0, 5.0),
    ]
    titles = ["(a) 线性变化", "(b) 随信号相位变化", "(c) 分段变化"]
    fig, axes = plt.subplots(
        2, 3, figsize=(7.3, 3.25), gridspec_kw={"height_ratios": [2.15, 1]}, sharex=True
    )
    fig.subplots_adjust(
        wspace=0.24, hspace=0.20, top=0.81, bottom=0.17, left=0.08, right=0.985
    )
    for i, s in enumerate(std):
        rng = np.random.default_rng(7010 + i + 1)
        rng.normal(size=10000)
        y = u + s * rng.normal(size=200)
        a, b = axes[:, i]
        a.fill_between(
            t, u - 1.95996398454 * s, u + 1.95996398454 * s, color="#a8c7ec", alpha=0.95
        )
        a.plot(t, u, color="#182e48", lw=0.8)
        a.scatter(t, y, c=ORANGE, s=1.6, alpha=0.7, edgecolors="none")
        a.set_title(titles[i], fontsize=9.5, fontweight="bold", color=BLUE, pad=3)
        a.set_ylim(-118, 118)
        a.set_yticks([-100, 0, 100])
        b.plot(
            t, s, color=BLUE, lw=1.25, drawstyle="steps-post" if i == 2 else "default"
        )
        b.set_ylim(-0.3, 6.7)
        b.set_yticks([0, 3, 6])
        b.set_xticks([0, 0.5, 1])
        b.set_xlabel("相位 t", fontsize=8.3, labelpad=1)
        for ax in (a, b):
            ax.grid(axis="y", color=GRIDL, lw=0.5)
            ax.tick_params(labelsize=7.6, pad=2)
        if i == 0:
            a.set_ylabel("观测值", fontsize=9)
            b.set_ylabel("标准差", fontsize=9)
        else:
            a.tick_params(labelleft=False)
            b.tick_params(labelleft=False)
    handles = [
        Line2D([], [], color="#182e48", lw=1, label="真实均值"),
        Line2D([], [], marker=".", linestyle="none", color=ORANGE, label="测试观测"),
        Patch(facecolor="#a8c7ec", label="真实 95% 预测区间"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.00),
        frameon=False,
        ncol=3,
        fontsize=8.5,
    )
    save(fig, loc, "fig2")


def trajectory(loc, df, bearing=False):
    df = df.sort_values("elapsed")
    t = df.elapsed.to_numpy()
    truth = df.y.to_numpy()
    palette = {"SREnbPI": BLUE, "EnbPI": "#655294", "MC": ORANGE, "DE": GREEN}
    names = {
        "SREnbPI": "SREnbPI",
        "EnbPI": "EnbPI",
        "MC": "MC Dropout",
        "DE": "Deep Ensemble",
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 3.9), sharex=True)
    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.115, top=0.85, wspace=0.26, hspace=0.38
    )
    axes = axes.flatten()
    mlist = ["SREnbPI", "MC", "DE"]
    lo = min(np.min(df[m + "_mid"] - df[m + "_half"]) for m in mlist)
    hi = max(np.max(df[m + "_mid"] + df[m + "_half"]) for m in mlist)
    ymin = min(0, lo) - 4
    ymax = max(np.max(truth), hi) + 5
    for i, (ax, m) in enumerate(zip(axes, mlist)):
        mid, half = df[m + "_mid"].to_numpy(), df[m + "_half"].to_numpy()
        ax.fill_between(t, mid - half, mid + half, color="#c2d7ef", alpha=0.91)
        ax.plot(t, mid, color=BLUE, lw=1.05)
        ax.plot(t, truth, color="#20262c", ls="--", lw=1.1)
        ax.set_ylim(ymin, ymax)
        ax.set_title(
            "(" + chr(97 + i) + ") " + names[m],
            loc="left",
            color=BLUE,
            fontweight="bold",
            fontsize=9.3,
            pad=2,
        )
    ax = axes[3]
    ax.set_title(
        "(d) 各方法的区间全宽",
        loc="left",
        color=BLUE,
        fontweight="bold",
        fontsize=9.3,
        pad=2,
    )
    for m, ls0 in zip(["SREnbPI", "EnbPI", "MC", "DE"], ["-", "--", ":", "-."]):
        ax.plot(
            t,
            2 * df[m + "_half"],
            color=palette[m],
            label=names[m],
            lw=1.15,
            linestyle=ls0,
        )
    ax.legend(
        loc="best",
        ncol=2,
        frameon=True,
        fontsize=6.0,
        columnspacing=0.7,
        handlelength=1.7,
        framealpha=0.88,
    )
    unit = "min" if bearing else "cycle"
    for i, ax in enumerate(axes):
        ax.grid(axis="y", color=GRIDL, lw=0.6)
        ax.tick_params(labelsize=7.5, pad=1.5)
        ax.set_ylabel(
            ("区间宽度" if i == 3 else "RUL") + " / " + unit, fontsize=8.1, labelpad=2
        )
        if i >= 2:
            ax.set_xlabel("已运行时间 / " + unit, fontsize=8.2, labelpad=1.5)
    handles = [
        Line2D([], [], color="#20262c", ls="--", label="真实 RUL"),
        Line2D([], [], color=BLUE, label="点预测"),
        Patch(facecolor="#c2d7ef", label="名义 95% 预测区间"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.991),
        ncol=3,
        frameon=False,
        fontsize=8.4,
    )
    save(fig, loc, "fig5" if bearing else "fig4")


def run(data, loc):
    loc.mkdir(parents=True, exist_ok=True)
    diagram(loc)
    simulation(loc)
    architecture(loc)
    fd = pd.read_csv(data / "fd_0_101.csv.gz")
    trajectory(loc, fd[(fd.id == 40) & (fd.subset == "trajectory")])
    bearing = pd.read_csv(data / "bearing_4_101.csv.gz")
    trajectory(loc, bearing, True)


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("data", type=Path)
    arg.add_argument("out", type=Path)
    a = arg.parse_args()
    run(a.data, a.out)
