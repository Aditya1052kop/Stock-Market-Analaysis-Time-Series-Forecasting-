# 📈 Stock Market Analysis Dashboard

Time Series Forecasting Dashboard for IT vs Petroleum sector stocks (NSE India).

---

## 🗂️ Folder Structure

```
stock_project/
│
├── app.py                     ← Main Streamlit dashboard (run this)
│
├── requirements.txt           ← All dependencies
│
├── data/                      ← Place all your CSV files here
│   ├── TCS-17-04-2025-17-04-2026.csv
│   ├── INFOSYS-17-04-2025-17-04-2026.csv
│   ├── HCLTECH-17-04-2025-17-04-2026.csv
│   ├── WIPRO-17-04-2025-17-04-2026.csv
│   ├── TECHMAHINDRA-17-04-2025-17-04-2026.csv
│   ├── RELIANCE-17-04-2025-17-04-2026.csv
│   ├── ONGC-17-04-2025-17-04-2026.csv
│   ├── IOC-17-04-2025-17-04-2026.csv
│   ├── BPCL-17-04-2025-17-04-2026.csv
│   └── HINDPETRO-17-04-2025-17-04-2026.csv
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py         ← CSV loading & cleaning
│   └── indicators.py          ← MA, Bollinger, RSI, MACD, Returns
│
├── models/
│   ├── __init__.py
│   └── forecasting.py         ← ARIMA model + stationarity test
│
└── assets/                    ← (optional) images, logos
```

---

## ⚙️ Setup Instructions

### Step 1 — Clone / create the project folder

```bash
mkdir stock_project
cd stock_project
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Add your data files

Copy all 10 CSV files into the `data/` folder.
The files must start with the stock name (e.g. `TCS`, `INFOSYS`, `HCLTECH`…).

### Step 4 — Run the dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 📊 Dashboard Features

| Feature | Description |
|---|---|
| KPI Cards | Current price, 52W high/low, avg price, volatility, return |
| Price Chart | Interactive close price with MA 7/20/50 overlay |
| Bollinger Bands | Dynamic support/resistance bands |
| RSI | Overbought (>70) / Oversold (<30) zones |
| MACD | Momentum crossover indicator |
| Volume | Daily trading volume bars |
| ARIMA Forecast | Configurable (p,d,q) with 95% confidence interval |
| Stationarity Test | ADF test results shown before forecast |
| Sector Comparison | Side-by-side IT vs Petroleum price charts |
| Cumulative Return | All 10 stocks normalised return chart |
| Correlation Heatmap | Return correlation across all stocks |
| Raw Data Export | Download processed CSV |

---

## 🧠 How ARIMA Works (Brief)

1. **ADF Test** checks if the price series is stationary
2. **d** = degree of differencing (usually 1 for stock prices)
3. **p** = autoregressive lag (looks back at past prices)
4. **q** = moving average error terms
5. Default order **(1, 1, 1)** works well for most stocks
6. The 95% confidence interval shows the uncertainty range

---

## 📁 CSV Format Expected

Your NSE CSV files should have columns:
`DATE, SERIES, OPEN, HIGH, LOW, PREV. CLOSE, LTP, CLOSE, VWAP, 52W H, 52W L, VOLUME, VALUE, NO. OF TRADES`

The loader handles comma-formatted numbers (e.g. `"2,501.90"`) automatically.

---

## 🚀 Future Improvements

- [ ] Add LSTM deep learning model
- [ ] Add Prophet forecasting
- [ ] Add candlestick chart (plotly)
- [ ] Add portfolio returns simulation
- [ ] Deploy on Streamlit Cloud
