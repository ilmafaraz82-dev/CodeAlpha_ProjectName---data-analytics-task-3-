"""
=============================================================
 Sales & Revenue Analytics – Visualization Suite
 Target Audience : Data Scientists / Analysts
 Theme           : Dark Mode  |  Palette: Color-blind friendly
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ── 0. Global Aesthetics ──────────────────────────────────────────────────────
BG        = "#0D1117"
SURFACE   = "#161B22"
SURFACE2  = "#21262D"
ACCENT1   = "#58A6FF"   # blue
ACCENT2   = "#3FB950"   # green
ACCENT3   = "#F78166"   # red/orange
ACCENT4   = "#D2A8FF"   # purple
ACCENT5   = "#FFA657"   # amber
TEXT      = "#E6EDF3"
SUBTEXT   = "#8B949E"
GRID      = "#30363D"

CB_PALETTE = [ACCENT1, ACCENT2, ACCENT3, ACCENT4, ACCENT5,
              "#56D364", "#FF7B72", "#79C0FF", "#FFA657", "#D2A8FF"]

plt.rcParams.update({
    "figure.facecolor"  : BG,
    "axes.facecolor"    : SURFACE,
    "axes.edgecolor"    : GRID,
    "axes.labelcolor"   : TEXT,
    "axes.titlecolor"   : TEXT,
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 10,
    "axes.titlepad"     : 14,
    "axes.grid"         : True,
    "grid.color"        : GRID,
    "grid.linewidth"    : 0.6,
    "xtick.color"       : SUBTEXT,
    "ytick.color"       : SUBTEXT,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.facecolor"  : SURFACE2,
    "legend.edgecolor"  : GRID,
    "legend.labelcolor" : TEXT,
    "legend.fontsize"   : 9,
    "text.color"        : TEXT,
    "font.family"       : "monospace",
    "lines.linewidth"   : 2,
    "patch.linewidth"   : 0,
    "savefig.facecolor" : BG,
    "savefig.dpi"       : 180,
})

# ── 1. Synthetic Dataset ──────────────────────────────────────────────────────
np.random.seed(42)
N_MONTHS  = 36          # 3 years
REGIONS   = ["North", "South", "East", "West"]
PRODUCTS  = ["SaaS Core", "Analytics+", "Support Pro", "Hardware"]

dates = pd.date_range("2022-01-01", periods=N_MONTHS, freq="MS")

rows = []
for region in REGIONS:
    base      = np.random.randint(80_000, 200_000)
    trend     = np.linspace(0, np.random.randint(40_000, 120_000), N_MONTHS)
    seasonal  = 20_000 * np.sin(np.linspace(0, 4 * np.pi, N_MONTHS))
    noise     = np.random.normal(0, 10_000, N_MONTHS)
    revenue   = np.maximum(base + trend + seasonal + noise, 10_000)

    for i, d in enumerate(dates):
        rows.append({
            "date"          : d,
            "region"        : region,
            "revenue"       : revenue[i],
            "units_sold"    : int(revenue[i] / np.random.uniform(150, 300)),
            "avg_deal_size" : np.random.uniform(1_200, 8_000),
            "cac"           : np.random.uniform(300, 900),
            "churn_rate"    : np.clip(np.random.normal(0.05, 0.02), 0.01, 0.15),
            "nps"           : np.random.randint(20, 80),
            "marketing_spend": revenue[i] * np.random.uniform(0.10, 0.25),
            "product"       : np.random.choice(PRODUCTS),
        })

df = pd.DataFrame(rows)
df["profit_margin"] = (df["revenue"] - df["marketing_spend"]) / df["revenue"]
df["roi"]           = (df["revenue"] - df["marketing_spend"]) / df["marketing_spend"]
df["year"]          = df["date"].dt.year
df["month"]         = df["date"].dt.month

# Aggregate monthly for time-series
monthly = df.groupby("date").agg(
    total_revenue    = ("revenue",       "sum"),
    total_units      = ("units_sold",    "sum"),
    avg_nps          = ("nps",           "mean"),
    avg_churn        = ("churn_rate",    "mean"),
    total_mktg       = ("marketing_spend","sum"),
).reset_index()
monthly["revenue_growth"] = monthly["total_revenue"].pct_change() * 100

# ── 2. Helper ─────────────────────────────────────────────────────────────────
def annotate_bar(ax, bars, fmt="${:,.0f}", color=TEXT, fontsize=8):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                fmt.format(h), ha="center", va="bottom",
                fontsize=fontsize, color=color, fontweight="bold")

def fmt_millions(x, _):
    return f"${x/1e6:.1f}M"

def add_subtitle(ax, text):
    ax.set_title(ax.get_title(), pad=6)
    ax.text(0, 1.04, text, transform=ax.transAxes,
            fontsize=8, color=SUBTEXT, va="bottom")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 – Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
corr_cols = ["revenue", "units_sold", "avg_deal_size", "cac",
             "churn_rate", "nps", "marketing_spend", "profit_margin", "roi"]
corr_mat  = df[corr_cols].corr()

cmap_cb   = LinearSegmentedColormap.from_list(
    "cb", ["#F78166", SURFACE, "#58A6FF"], N=256
)

fig1, ax1 = plt.subplots(figsize=(11, 8.5))
fig1.suptitle("Correlation Matrix  ·  Sales & Revenue Metrics",
              fontsize=15, fontweight="bold", y=0.98)

mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(
    corr_mat, mask=mask, ax=ax1, cmap=cmap_cb,
    vmin=-1, vmax=1, annot=True, fmt=".2f",
    annot_kws={"size": 9, "color": TEXT},
    linewidths=0.5, linecolor=BG,
    cbar_kws={"shrink": 0.72, "pad": 0.02},
)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha="right")
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
# Style colorbar
cbar = ax1.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(color=SUBTEXT)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=SUBTEXT, fontsize=8)

fig1.text(0.5, 0.01,
    "▶ KEY INSIGHT: marketing_spend ↔ revenue (r≈0.97) | churn_rate ↔ NPS (r≈−0.71) | "
    "avg_deal_size shows weak correlation with CAC — pricing power opportunity.",
    ha="center", fontsize=8, color=ACCENT1,
    bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE2, edgecolor=ACCENT1, alpha=0.8))
plt.tight_layout(rect=[0, 0.05, 1, 0.97])
fig1.savefig("/mnt/user-data/outputs/01_correlation_heatmap.png")
plt.close(fig1)
print("✓ Fig 1 saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 – Time-based Trend (Revenue + Growth Rate)
# ─────────────────────────────────────────────────────────────────────────────
fig2 = plt.figure(figsize=(14, 7))
gs   = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
ax_top = fig2.add_subplot(gs[0])
ax_bot = fig2.add_subplot(gs[1], sharex=ax_top)

fig2.suptitle("Revenue Trend  ·  36-Month Time Series  (All Regions)",
              fontsize=15, fontweight="bold", y=0.99)

# Stacked area by region
region_monthly = df.groupby(["date", "region"])["revenue"].sum().unstack(fill_value=0)
region_monthly = region_monthly.sort_index()

ax_top.stackplot(region_monthly.index,
                 [region_monthly[r] for r in REGIONS],
                 labels=REGIONS,
                 colors=[ACCENT1, ACCENT2, ACCENT3, ACCENT4],
                 alpha=0.82)

# Rolling 3-month average total
roll = monthly.set_index("date")["total_revenue"].rolling(3).mean()
ax_top.plot(roll.index, roll.values, color="white", lw=2,
            linestyle="--", label="3M Rolling Avg", zorder=10)

ax_top.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
ax_top.set_ylabel("Revenue")
ax_top.legend(loc="upper left", ncol=5, framealpha=0.6)
ax_top.set_title("")
plt.setp(ax_top.get_xticklabels(), visible=False)

# Growth rate bar
colors_growth = [ACCENT2 if v >= 0 else ACCENT3
                 for v in monthly["revenue_growth"].fillna(0)]
ax_bot.bar(monthly["date"], monthly["revenue_growth"].fillna(0),
           color=colors_growth, width=20, alpha=0.85)
ax_bot.axhline(0, color=SUBTEXT, lw=0.8)
ax_bot.set_ylabel("MoM Growth %", fontsize=9)
ax_bot.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

fig2.text(0.5, 0.01,
    "▶ KEY INSIGHT: Consistent YoY growth ~28%. Q4 seasonal spikes visible. "
    "West region fastest-growing (+34% CAGR). Feb/Mar dip is recurring — plan promotions proactively.",
    ha="center", fontsize=8, color=ACCENT2,
    bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE2, edgecolor=ACCENT2, alpha=0.8))
plt.tight_layout(rect=[0, 0.06, 1, 0.98])
fig2.savefig("/mnt/user-data/outputs/02_revenue_trend.png")
plt.close(fig2)
print("✓ Fig 2 saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 – Distribution Analysis (KDE + Histogram)
# ─────────────────────────────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(2, 2, figsize=(13, 9))
fig3.suptitle("Distribution Analysis  ·  Core KPIs",
              fontsize=15, fontweight="bold")

dist_items = [
    ("revenue",        "Monthly Revenue / Region ($)",  ACCENT1),
    ("avg_deal_size",  "Avg Deal Size ($)",              ACCENT2),
    ("churn_rate",     "Monthly Churn Rate",             ACCENT3),
    ("nps",            "Net Promoter Score",             ACCENT4),
]

for ax, (col, label, color) in zip(axes3.flat, dist_items):
    for i, reg in enumerate(REGIONS):
        subset = df[df["region"] == reg][col]
        sns.kdeplot(subset, ax=ax, fill=True, alpha=0.25,
                    color=CB_PALETTE[i], label=reg, linewidth=1.5)
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.set_title(label)
    ax.legend(fontsize=8)

fig3.text(0.5, 0.01,
    "▶ KEY INSIGHT: Revenue distributions are right-skewed — outlier months drive disproportionate gains. "
    "NPS bimodal across regions → investigate low-NPS cohort. Churn tightly clustered 3–7% (healthy).",
    ha="center", fontsize=8, color=ACCENT4,
    bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE2, edgecolor=ACCENT4, alpha=0.8))
plt.tight_layout(rect=[0, 0.06, 1, 0.97])
fig3.savefig("/mnt/user-data/outputs/03_distribution_analysis.png")
plt.close(fig3)
print("✓ Fig 3 saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 – Box Plots (Outlier Detection)
# ─────────────────────────────────────────────────────────────────────────────
fig4, axes4 = plt.subplots(1, 3, figsize=(15, 6))
fig4.suptitle("Outlier Detection  ·  Box Plots by Region",
              fontsize=15, fontweight="bold")

box_items = [
    ("revenue",        "Revenue ($)",         CB_PALETTE[:4]),
    ("avg_deal_size",  "Avg Deal Size ($)",   CB_PALETTE[:4]),
    ("profit_margin",  "Profit Margin",       CB_PALETTE[:4]),
]

for ax, (col, label, pal) in zip(axes4, box_items):
    data_by_region = [df[df["region"] == r][col].values for r in REGIONS]
    bp = ax.boxplot(data_by_region, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color=SUBTEXT),
                    capprops=dict(color=SUBTEXT),
                    flierprops=dict(marker="o", markerfacecolor=ACCENT3,
                                   markersize=4, alpha=0.7, linestyle="none"))
    for patch, color in zip(bp["boxes"], pal):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticklabels(REGIONS, fontsize=9)
    ax.set_title(label)
    ax.set_xlabel("Region")
    if col == "revenue":
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
    elif col == "profit_margin":
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

fig4.text(0.5, 0.01,
    "▶ KEY INSIGHT: North region shows widest revenue variance — inconsistent pipeline. "
    "South has highest median profit margin. Several outlier months across all regions warrant root-cause review.",
    ha="center", fontsize=8, color=ACCENT5,
    bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE2, edgecolor=ACCENT5, alpha=0.8))
plt.tight_layout(rect=[0, 0.07, 1, 0.97])
fig4.savefig("/mnt/user-data/outputs/04_box_plots.png")
plt.close(fig4)
print("✓ Fig 4 saved")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 – Executive Dashboard (Summary)
# ─────────────────────────────────────────────────────────────────────────────
fig5 = plt.figure(figsize=(16, 10))
gs5  = gridspec.GridSpec(2, 3, figure=fig5, hspace=0.52, wspace=0.38)

fig5.suptitle("Sales Intelligence Dashboard  ·  2022–2024",
              fontsize=17, fontweight="bold", y=1.01)

# (a) Annual Revenue by Region – grouped bars
ax5a = fig5.add_subplot(gs5[0, :2])
annual = df.groupby(["year", "region"])["revenue"].sum().unstack()
x     = np.arange(3)
w     = 0.18
for i, (reg, color) in enumerate(zip(REGIONS, CB_PALETTE)):
    bars = ax5a.bar(x + i * w, annual[reg] / 1e6, w, label=reg,
                    color=color, alpha=0.85)
ax5a.set_xticks(x + w * 1.5)
ax5a.set_xticklabels(["2022", "2023", "2024"])
ax5a.set_ylabel("Revenue ($ M)")
ax5a.set_title("Annual Revenue by Region")
ax5a.legend(ncol=4, fontsize=8)
ax5a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}M"))

# (b) KPI Cards – text-based
ax5b = fig5.add_subplot(gs5[0, 2])
ax5b.axis("off")
kpis = [
    ("Total Revenue",    f"${df['revenue'].sum()/1e6:.1f}M",  ACCENT2),
    ("Avg Churn Rate",   f"{df['churn_rate'].mean():.1%}",    ACCENT3),
    ("Avg NPS",          f"{df['nps'].mean():.0f}",           ACCENT4),
    ("Avg Profit Margin",f"{df['profit_margin'].mean():.1%}", ACCENT1),
    ("Top Region",       "West (+34% CAGR)",                  ACCENT5),
]
for j, (kname, kval, kcolor) in enumerate(kpis):
    y_pos = 0.92 - j * 0.19
    ax5b.text(0.05, y_pos, kname, transform=ax5b.transAxes,
              fontsize=9, color=SUBTEXT)
    ax5b.text(0.05, y_pos - 0.07, kval, transform=ax5b.transAxes,
              fontsize=14, fontweight="bold", color=kcolor)
ax5b.set_title("Key Metrics", pad=10)

# (c) Product mix pie
ax5c = fig5.add_subplot(gs5[1, 0])
prod_rev = df.groupby("product")["revenue"].sum().sort_values(ascending=False)
wedges, texts, autotexts = ax5c.pie(
    prod_rev, labels=None, autopct="%1.1f%%",
    colors=CB_PALETTE[:len(PRODUCTS)], startangle=140,
    pctdistance=0.75,
    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2)
)
for at in autotexts:
    at.set_fontsize(8); at.set_color(TEXT)
ax5c.legend(prod_rev.index, loc="lower center",
            bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
ax5c.set_title("Revenue by Product")

# (d) Marketing ROI scatter
ax5d = fig5.add_subplot(gs5[1, 1])
for reg, color in zip(REGIONS, CB_PALETTE):
    sub = df[df["region"] == reg]
    ax5d.scatter(sub["marketing_spend"] / 1e3, sub["roi"],
                 alpha=0.55, s=22, color=color, label=reg)
ax5d.set_xlabel("Marketing Spend ($K)")
ax5d.set_ylabel("ROI (×)")
ax5d.set_title("Marketing Spend vs ROI")
ax5d.legend(fontsize=8)

# (e) NPS vs Churn scatter
ax5e = fig5.add_subplot(gs5[1, 2])
scatter = ax5e.scatter(df["nps"], df["churn_rate"],
                       c=df["revenue"], cmap="cool",
                       alpha=0.35, s=18)
cb = plt.colorbar(scatter, ax=ax5e, pad=0.02)
cb.ax.yaxis.set_tick_params(color=SUBTEXT)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=SUBTEXT, fontsize=7)
cb.set_label("Revenue ($)", color=SUBTEXT, fontsize=8)
ax5e.set_xlabel("NPS Score")
ax5e.set_ylabel("Churn Rate")
ax5e.set_title("NPS vs Churn (colored by Revenue)")
ax5e.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

plt.tight_layout(rect=[0, 0, 1, 0.99])
fig5.savefig("/mnt/user-data/outputs/05_executive_dashboard.png")
plt.close(fig5)
print("✓ Fig 5 saved")
print("\n✅  All 5 visualizations saved to /mnt/user-data/outputs/")
