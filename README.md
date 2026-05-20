# QuantEdge v2.1

AI-powered trading strategy backtester with multi-strategy support and LLM analysis.

## Project Structure

```
quantedge/
├── api/                    ← Vercel serverless functions
│   ├── backtest.py         ← POST /api/backtest
│   ├── analyze.py          ← POST /api/analyze
│   ├── optimize.py         ← POST /api/optimize
│   ├── parse_strategy.py   ← POST /api/parse-strategy
│   └── status.py           ← GET  /api/status
├── public/
│   └── index.html          ← Frontend
├── backtest_engine.py      ← Core backtest logic (shared)
├── app.py                  ← Flask server (local dev only)
├── requirements.txt
├── vercel.json
└── .env.example
```

## Vercel Deployment

### 1. Deploy to Vercel

```bash
npm i -g vercel
vercel --prod
```

### 2. Add Environment Variable

In Vercel Dashboard → Your Project → Settings → Environment Variables:

```
GROQ_API_KEY = your_key_here
```

Get a **free** Groq API key at: https://console.groq.com

### 3. Done!

All API endpoints will be live at `https://your-project.vercel.app/api/*`

---

## Local Development

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
python app.py
# Open http://localhost:5000
```

---

## Fixes in v2.1

- **"No data for AAPL" fix**: `yfinance` now uses proper `start`/`end` date params instead of `period=` which fails on Vercel
- **Vercel structure fix**: API functions moved to `/api/` folder as required by Vercel
- **AI analyze fix**: Correct prompt and response format
- **AI optimize fix**: Response now returns `suggestions[]` array matching frontend expectations  
- **Parse-strategy fix**: Correct request/response field names
- **`requirements.txt`**: Added so Vercel installs Python dependencies
