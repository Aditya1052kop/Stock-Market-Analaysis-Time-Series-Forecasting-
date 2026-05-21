"""
app.py  —  Stock Market Analysis Dashboard
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np
import sys, os

# ── make local packages importable ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader  import load_data, load_multiple, STOCK_FILE_MAP
from utils.indicators   import (add_moving_averages, add_bollinger_bands,
                                 add_rsi, add_macd,
                                 add_daily_returns, summary_stats)
from models.forecasting import run_arima, check_stationarity, auto_order

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .metric-card {
        background: #1e293b; border-radius: 12px;
        padding: 16px; text-align: center;
        border: 1px solid #334155;
    }
    .metric-label { font-size: 12px; color: #94a3b8; }
    .metric-value { font-size: 24px; font-weight: 700; color: #38bdf8; }
    .metric-sub   { font-size: 13px; color: #64748b; }
    .section-title { font-size: 20px; font-weight: 600;
                     border-left: 4px solid #38bdf8;
                     padding-left: 10px; margin: 24px 0 12px; }
</style>
""", unsafe_allow_html=True)

# ── sidebar ─────────────────────────────────────────────────────────────────
STOCKS = list(STOCK_FILE_MAP.keys())
IT_STOCKS  = ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM"]
OIL_STOCKS = ["RELIANCE", "ONGC", "IOC", "BPCL", "HINDPETRO"]

st.sidebar.title("📊 Stock Dashboard")
st.sidebar.markdown("---")

stock = st.sidebar.selectbox("🔍 Select Stock", STOCKS)

st.sidebar.markdown("**Chart Options**")
show_ma   = st.sidebar.checkbox("Moving Averages", value=True)
show_bb   = st.sidebar.checkbox("Bollinger Bands", value=False)
show_rsi  = st.sidebar.checkbox("RSI Indicator",   value=True)
show_macd = st.sidebar.checkbox("MACD",            value=False)
show_vol  = st.sidebar.checkbox("Volume",          value=True)

st.sidebar.markdown("**Forecast**")
forecast_days = st.sidebar.slider("Days to Forecast", 5, 30, 10)

# ── ARIMA order fixed internally (p=1, d=1, q=1) ───────────────────────────
arima_p = 1
arima_d = 1
arima_q = 1

# ── load data ───────────────────────────────────────────────────────────────
df_raw = load_data(stock)
if df_raw is None:
    st.error(f"❌ Could not load data for **{stock}**. "
             "Make sure the CSV is in the `data/` folder.")
    st.stop()

# Enrich with indicators
df = add_moving_averages(df_raw)
df = add_bollinger_bands(df)
df = add_rsi(df)
df = add_macd(df)
df = add_daily_returns(df)

stats = summary_stats(df)

# ── header ───────────────────────────────────────────────────────────────────
sector = "💻 IT Sector" if stock in IT_STOCKS else "🛢️ Petroleum Sector"
st.title(f"📈 {stock} — {sector}")
st.caption("NSE Daily Data  |  Apr 2025 – Apr 2026  |  ARIMA Forecasting")
st.markdown("---")

# ── KPI cards ────────────────────────────────────────────────────────────────
cols = st.columns(len(stats))
icons = ["💰", "⬆️", "⬇️", "📊", "📉", "📅", "🚀"]
for col, (k, v), icon in zip(cols, stats.items(), icons):
    color = "#22c55e" if isinstance(v, float) and v >= 0 else "#f87171"
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {k}</div>
        <div class="metric-value" style="color:{color};">{v}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── price chart ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Price Chart</div>', unsafe_allow_html=True)

plot_rows = 1
if show_rsi:  plot_rows += 1
if show_macd: plot_rows += 1
if show_vol:  plot_rows += 1

heights = [3] + [1] * (plot_rows - 1)
fig = plt.figure(figsize=(14, 4 * plot_rows), facecolor="#0f172a")
gs  = gridspec.GridSpec(plot_rows, 1, figure=fig,
                        height_ratios=heights, hspace=0.08)

ax_price = fig.add_subplot(gs[0])
ax_price.plot(df.index, df["Close"], color="#38bdf8", linewidth=1.5,
              label="Close", zorder=3)

if show_ma:
    ax_price.plot(df.index, df["MA_7"],  color="#facc15", linewidth=1,
                  linestyle="--", label="MA 7", alpha=0.85)
    ax_price.plot(df.index, df["MA_20"], color="#fb923c", linewidth=1,
                  linestyle="--", label="MA 20", alpha=0.85)
    ax_price.plot(df.index, df["MA_50"], color="#a78bfa", linewidth=1,
                  linestyle="--", label="MA 50", alpha=0.85)

if show_bb:
    ax_price.fill_between(df.index, df["BB_lower"], df["BB_upper"],
                          color="#38bdf8", alpha=0.08, label="Bollinger Band")
    ax_price.plot(df.index, df["BB_upper"], color="#38bdf8",
                  linewidth=0.6, linestyle=":")
    ax_price.plot(df.index, df["BB_lower"], color="#38bdf8",
                  linewidth=0.6, linestyle=":")

ax_price.set_facecolor("#0f172a")
ax_price.tick_params(colors="#94a3b8", labelbottom=False)
ax_price.set_ylabel("Price (₹)", color="#94a3b8")
ax_price.spines[:].set_color("#334155")
ax_price.legend(facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=8)
ax_price.grid(color="#1e293b", linewidth=0.5)
ax_price.set_title(f"{stock} — Closing Price", color="#e2e8f0", pad=8)

row = 1

if show_vol and "Volume" in df.columns:
    ax_vol = fig.add_subplot(gs[row], sharex=ax_price); row += 1
    ax_vol.bar(df.index, df["Volume"], color="#475569", width=1)
    ax_vol.set_facecolor("#0f172a")
    ax_vol.set_ylabel("Volume", color="#94a3b8", fontsize=8)
    ax_vol.tick_params(colors="#94a3b8", labelbottom=False)
    ax_vol.spines[:].set_color("#334155")
    ax_vol.grid(color="#1e293b", linewidth=0.5)

if show_rsi:
    ax_rsi = fig.add_subplot(gs[row], sharex=ax_price); row += 1
    ax_rsi.plot(df.index, df["RSI"], color="#34d399", linewidth=1)
    ax_rsi.axhline(70, color="#f87171", linewidth=0.8, linestyle="--")
    ax_rsi.axhline(30, color="#facc15", linewidth=0.8, linestyle="--")
    ax_rsi.fill_between(df.index, df["RSI"], 70,
                        where=(df["RSI"] >= 70), color="#f87171", alpha=0.2)
    ax_rsi.fill_between(df.index, df["RSI"], 30,
                        where=(df["RSI"] <= 30), color="#292000", alpha=0.2)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_facecolor("#0f172a")
    ax_rsi.set_ylabel("RSI", color="#94a3b8", fontsize=8)
    ax_rsi.tick_params(colors="#94a3b8", labelbottom=False)
    ax_rsi.spines[:].set_color("#334155")
    ax_rsi.grid(color="#1e293b", linewidth=0.5)

if show_macd:
    ax_macd = fig.add_subplot(gs[row], sharex=ax_price); row += 1
    ax_macd.plot(df.index, df["MACD"],     color="#38bdf8", linewidth=1, label="MACD")
    ax_macd.plot(df.index, df["MACD_sig"], color="#f472b6", linewidth=1, label="Signal")
    colors = ["#22c55e" if v >= 0 else "#f87171" for v in df["MACD_hist"]]
    ax_macd.bar(df.index, df["MACD_hist"], color=colors, width=1, alpha=0.6)
    ax_macd.set_facecolor("#0f172a")
    ax_macd.set_ylabel("MACD", color="#94a3b8", fontsize=8)
    ax_macd.tick_params(colors="#94a3b8")
    ax_macd.spines[:].set_color("#334155")
    ax_macd.legend(facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=7)
    ax_macd.grid(color="#1e293b", linewidth=0.5)

# format x-axis on last subplot
last_ax = fig.axes[-1]
last_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
last_ax.tick_params(axis="x", colors="#94a3b8", labelbottom=True)

plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close()

# ── ARIMA Forecast ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">ARIMA Forecast</div>',
            unsafe_allow_html=True)

if len(df) < 30:
    st.warning("⚠️ Not enough data for reliable forecasting (need ≥ 30 rows).")
else:
    stat_result = check_stationarity(df["Close"])
    c1, c2, c3 = st.columns(3)
    c1.metric("ADF Statistic",  stat_result["adf_stat"])
    c2.metric("p-value",        stat_result["p_value"])
    c3.metric("Stationary?",    "✅ Yes" if stat_result["is_stationary"] else "❌ No")

    with st.spinner("Running ARIMA…"):
        result = run_arima(df["Close"],
                           order=(arima_p, arima_d, arima_q),
                           steps=forecast_days)

    if not result["success"]:
        st.error(f"ARIMA failed: {result['error']}")
    else:
        fc   = result["forecast_values"]
        ci   = result["conf_int"]

        # Forecast plot
        fig2, ax2 = plt.subplots(figsize=(14, 4), facecolor="#0f172a")
        ax2.plot(df.index[-60:], df["Close"].iloc[-60:],
                 color="#38bdf8", label="Actual (last 60 days)", linewidth=1.5)
        ax2.plot(fc.index, fc,
                 color="#f472b6", label="Forecast", linewidth=2,
                 linestyle="--", marker="o", markersize=4)
        ax2.fill_between(fc.index, ci.iloc[:, 0], ci.iloc[:, 1],
                         color="#f472b6", alpha=0.15, label="95% CI")
        ax2.set_facecolor("#0f172a")
        ax2.spines[:].set_color("#334155")
        ax2.tick_params(colors="#94a3b8")
        ax2.set_ylabel("Price (₹)", color="#94a3b8")
        ax2.legend(facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=9)
        ax2.grid(color="#1e293b", linewidth=0.5)
        ax2.set_title(f"{stock} — {forecast_days}-Day ARIMA Forecast",
                      color="#e2e8f0")
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        # Forecast table
        fc_df = pd.DataFrame({
            "Date":         fc.index.strftime("%d %b %Y"),
            "Forecast (₹)": fc.round(2).values,
            "Lower 95%":    ci.iloc[:, 0].round(2).values,
            "Upper 95%":    ci.iloc[:, 1].round(2).values,
        })
        st.dataframe(fc_df, use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        col_a.metric("AIC", result["aic"])
        col_b.metric("BIC", result["bic"])

# ── Sector Comparison ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Sector Comparison</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💻 IT Stocks", "🛢️ Petroleum Stocks",
                             "📊 Cumulative Return"])

def _comparison_plot(stock_list, title):
    dfs = load_multiple(stock_list)
    if not dfs:
        st.warning("No data found for these stocks.")
        return
    fig, ax = plt.subplots(figsize=(13, 4), facecolor="#0f172a")
    palette = ["#38bdf8","#34d399","#facc15","#fb923c","#a78bfa"]
    for (name, sdf), color in zip(dfs.items(), palette):
        ax.plot(sdf.index, sdf["Close"], label=name, color=color, linewidth=1.3)
    ax.set_facecolor("#0f172a")
    ax.spines[:].set_color("#334155")
    ax.tick_params(colors="#94a3b8")
    ax.set_ylabel("Price (₹)", color="#94a3b8")
    ax.legend(facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=9)
    ax.grid(color="#1e293b", linewidth=0.5)
    ax.set_title(title, color="#e2e8f0")
    st.pyplot(fig, use_container_width=True)
    plt.close()

with tab1:
    _comparison_plot(IT_STOCKS, "IT Sector — Closing Prices")

with tab2:
    _comparison_plot(OIL_STOCKS, "Petroleum Sector — Closing Prices")

with tab3:
    all_dfs = load_multiple(STOCKS)
    fig3, ax3 = plt.subplots(figsize=(13, 4), facecolor="#0f172a")
    palette = ["#38bdf8","#34d399","#facc15","#fb923c","#a78bfa",
               "#f472b6","#818cf8","#2dd4bf","#fbbf24","#4ade80"]
    for (name, sdf), color in zip(all_dfs.items(), palette):
        cum = (sdf["Close"] / sdf["Close"].iloc[0] - 1) * 100
        ax3.plot(sdf.index, cum, label=name, color=color, linewidth=1.2)
    ax3.axhline(0, color="#475569", linewidth=0.8, linestyle="--")
    ax3.set_facecolor("#0f172a")
    ax3.spines[:].set_color("#334155")
    ax3.tick_params(colors="#94a3b8")
    ax3.set_ylabel("Cumulative Return (%)", color="#94a3b8")
    ax3.legend(facecolor="#1e293b", labelcolor="#e2e8f0", fontsize=8,
               ncol=2)
    ax3.grid(color="#1e293b", linewidth=0.5)
    ax3.set_title("All Stocks — Cumulative Return (%)", color="#e2e8f0")
    st.pyplot(fig3, use_container_width=True)
    plt.close()

# ── Correlation Heatmap ──────────────────────────────────────────────────────
st.markdown('<div class="section-title">Correlation Heatmap</div>',
            unsafe_allow_html=True)

all_close = {}
for name, sdf in load_multiple(STOCKS).items():
    all_close[name] = sdf["Close"]

corr_df = pd.DataFrame(all_close).corr()

fig4, ax4 = plt.subplots(figsize=(10, 7), facecolor="#0f172a")
im = ax4.imshow(corr_df.values, cmap="RdYlGn", vmin=-1, vmax=1)
ax4.set_xticks(range(len(corr_df.columns)))
ax4.set_yticks(range(len(corr_df.index)))
ax4.set_xticklabels(corr_df.columns, rotation=45, ha="right", color="#e2e8f0")
ax4.set_yticklabels(corr_df.index, color="#e2e8f0")
ax4.set_facecolor("#0f172a")
for i in range(len(corr_df)):
    for j in range(len(corr_df.columns)):
        ax4.text(j, i, f"{corr_df.values[i, j]:.2f}",
                 ha="center", va="center", fontsize=8,
                 color="black" if abs(corr_df.values[i, j]) > 0.5 else "#e2e8f0")
plt.colorbar(im, ax=ax4)
ax4.set_title("Stock Return Correlation Matrix", color="#e2e8f0", pad=12)
fig4.patch.set_facecolor("#0f172a")
plt.tight_layout()
st.pyplot(fig4, use_container_width=True)
plt.close()

# ── Raw Data ─────────────────────────────────────────────────────────────────
with st.expander("📂 View Raw Dataset"):
    st.dataframe(df[["Open","High","Low","Close","Volume"]].sort_index(ascending=False),
                 use_container_width=True)
    csv = df.to_csv().encode("utf-8")
    st.download_button("⬇️ Download CSV", csv,
                       file_name=f"{stock}_processed.csv",
                       mime="text/csv")

# ── footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("📌 Built with Streamlit · ARIMA via statsmodels · NSE data")