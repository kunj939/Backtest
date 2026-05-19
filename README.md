# ◆ QuantEdge — Algorithmic Strategy Backtester

> A mini hedge-fund research prototype. MA Crossover strategy engine with AI-powered quant analysis, real market data, and institutional-grade metrics.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Vercel](https://img.shields.io/badge/Deploy-Vercel-black?style=flat-square&logo=vercel)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🚀 Live Demo

Deploy to Vercel in one click — see **Deploy** section below.

---

## 📌 Project Overview

QuantEdge is a **production-grade backtesting prototype** that simulates a real quant research workflow:

1. Fetches live OHLCV data via **yfinance**
2. Computes **Moving Average Crossover** signals (Golden Cross / Death Cross)
3. Simulates portfolio performance with **log-return arithmetic**
4. Calculates **institutional metrics** (Sharpe, Calmar, Max Drawdown, CAGR)
5. Generates **3-panel research charts** (Price + Signals, Equity Curve, Drawdown)
6. Sends results to **Claude (Anthropic)** for AI-powered quant interpretation

---

## 📐 Strategy Logic

| Signal | Condition | Interpretation |
|--------|-----------|----------------|
| **BUY ▲** | MA-20 crosses **above** MA-50 | Golden Cross — bullish momentum |
| **SELL ▼** | MA-20 crosses **below** MA-50 | Death Cross — bearish reversal |
| **HOLD** | No crossover | Maintain current position |

**Return calculation** uses log-returns for compounding accuracy:

```
Strategy Return = log(Close_t / Close_{t-1}) × Position_{t-1}
```

---

## 📊 Performance Metrics

| Metric | Description |
|--------|-------------|
| Total Return | Cumulative strategy P&L (%) |
| CAGR | Compound Annual Growth Rate |
| Sharpe Ratio | Annualized excess return / volatility (rf=5%) |
| Max Drawdown | Largest peak-to-trough decline |
| Win Rate | % of completed trades that were profitable |
| Calmar Ratio | CAGR / |Max Drawdown| — return per unit of drawdown risk |
| Volatility | Annualized standard deviation of daily returns |

---

## 🗂️ Project Structure

```
quant-backtest/
├── api/
│   ├── backtest.py       # Serverless: runs full backtest pipeline
│   └── analyze.py        # Serverless: Claude AI quant analysis
├── public/
│   └── index.html        # Frontend dashboard (vanilla JS)
├── src/
│   └── backtest.py       # Core engine: data → signals → metrics → charts
├── requirements.txt
├── vercel.json
└── README.md
```

---

## 🛠️ Technologies

| Layer | Tech |
|-------|------|
| Data | yfinance (Yahoo Finance) |
| Compute | pandas, numpy |
| Visualization | matplotlib (base64 PNG) |
| AI Analysis | Anthropic Claude API |
| Backend | Python serverless (Vercel) |
| Frontend | Vanilla HTML/CSS/JS |
| Deploy | Vercel |

---

## ⚡ Deploy to Vercel

### 1. Clone & install

```bash
git clone <your-repo>
cd quant-backtest
pip install -r requirements.txt
```

### 2. Set environment variable

In Vercel dashboard → Settings → Environment Variables:

```
ANTHROPIC_API_KEY = sk-ant-...
```

Or create a `.env` file locally:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Deploy

```bash
npm i -g vercel
vercel --prod
```

### 4. Local test (Python only)

```bash
cd src
python backtest.py
```

---

## 🔬 Run Locally (Full Stack)

```bash
# Install Vercel CLI
npm i -g vercel

# Run dev server
vercel dev
```

Open `http://localhost:3000`

---

## 🔮 Future Extensions (Research Roadmap)

### 1. ML-Based Signal Prediction
Replace the heuristic MA crossover with a supervised classifier:
- Features: OHLCV ratios, RSI, MACD, Bollinger Bands, sector momentum
- Models: XGBoost, LightGBM, or LSTM sequence model
- Target: forward 5-day return sign (binary classification)

### 2. Reinforcement Learning Agent
Train a PPO/SAC agent to learn position sizing and timing:
- Environment: gym-compatible market simulator
- State: rolling window of normalized OHLCV + indicators
- Reward: risk-adjusted PnL (Sharpe-scaled)

### 3. Multi-Strategy Portfolio Optimization
Combine multiple uncorrelated strategies:
- MA Crossover + RSI Mean-Reversion + Volatility Breakout
- Allocation via Markowitz MPT or Risk Parity
- Walk-forward optimization to prevent overfitting

### 4. Real-Time Execution Engine
- Connect Alpaca Markets or Interactive Brokers API
- Live paper trading with WebSocket price feeds
- Telegram/Slack alerts on signal triggers
- Drawdown circuit breakers for risk management

### 5. Alternative Data Integration
- Earnings sentiment via SEC EDGAR NLP
- Options flow (unusual activity screening)
- Crypto on-chain metrics (exchange flows, whale wallets)

---

## ⚠️ Disclaimer

This project is for **educational and demonstration purposes only**. It does not constitute financial advice. Past performance does not guarantee future results.

---

## 👤 Author

Built as a quant research portfolio project.  
Inspired by institutional research workflows at leading hedge funds and prop trading firms.

---

*QuantEdge Backtest Engine v1.0 · MIT License*
