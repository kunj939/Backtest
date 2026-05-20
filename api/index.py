"""
QuantEdge API — Single Flask entry point for Vercel
Vercel looks for a top-level 'app' object in this file.
ALL /api/* routes handled here.
"""
import os, sys, json, urllib.request, urllib.error
from flask import Flask, request, jsonify, make_response

# Add parent dir so backtest_engine.py is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backtest_engine import run_full_backtest

# ── Flask app — Vercel finds this 'app' object ────────────────────────────────
app = Flask(__name__)

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


# ── CORS ──────────────────────────────────────────────────────────────────────
def cors(resp):
    resp.headers['Access-Control-Allow-Origin']  = '*'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return resp

@app.before_request
def handle_preflight():
    if request.method == 'OPTIONS':
        resp = make_response()
        cors(resp)
        return resp, 200


# ── STATUS ────────────────────────────────────────────────────────────────────
@app.route('/api/status', methods=['GET'])
def status():
    key = os.environ.get('GROQ_API_KEY', '')
    return cors(jsonify({
        "status": "running", "server": "QuantEdge v2.1",
        "groq_key_loaded": bool(key), "model": GROQ_MODEL
    }))


# ── BACKTEST ──────────────────────────────────────────────────────────────────
@app.route('/api/backtest', methods=['POST', 'OPTIONS'])
def backtest():
    data = request.get_json(silent=True) or {}
    try:
        ticker   = str(data.get('ticker',   'AAPL')).upper().strip()
        start    = str(data.get('start',    '2020-01-01'))
        capital  = float(data.get('capital', 100_000))
        strategy = str(data.get('strategy', 'ma_crossover'))

        params = {
            'short_ma':       int(data.get('short_ma',       20)),
            'long_ma':        int(data.get('long_ma',        50)),
            'rsi_period':     int(data.get('rsi_period',     14)),
            'rsi_overbought': int(data.get('rsi_overbought', 70)),
            'rsi_oversold':   int(data.get('rsi_oversold',   30)),
            'macd_fast':      int(data.get('macd_fast',      12)),
            'macd_slow':      int(data.get('macd_slow',      26)),
            'macd_signal':    int(data.get('macd_signal',     9)),
            'bb_period':      int(data.get('bb_period',      20)),
            'bb_std':       float(data.get('bb_std',        2.0)),
        }
        if strategy == 'custom' and data.get('custom_rules'):
            params['custom_rules'] = data['custom_rules']

        if params['short_ma'] >= params['long_ma']:
            return cors(jsonify({"error": "short_ma must be less than long_ma"})), 400
        if capital < 1000:
            return cors(jsonify({"error": "Minimum capital is $1,000"})), 400

        ohlcv = data.get('ohlcv')  # pre-fetched from frontend
        result = run_full_backtest(
            ticker=ticker, start=start, capital=capital,
            strategy=strategy, params=params,
            ohlcv=ohlcv
        )
        return cors(jsonify(result))

    except Exception as e:
        print(f"BACKTEST ERROR: {e}")
        return cors(jsonify({"error": str(e)})), 500


# ── GROQ HELPER ───────────────────────────────────────────────────────────────
def groq_chat(prompt, system=None, max_tokens=800, temperature=0.3):
    key = os.environ.get('GROQ_API_KEY', '')
    if not key:
        raise Exception("GROQ_API_KEY not set. Add it in Vercel → Settings → Environment Variables.")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": GROQ_MODEL, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


# ── ANALYZE ───────────────────────────────────────────────────────────────────
@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    data    = request.get_json(silent=True) or {}
    metrics = data.get('metrics', {})
    ticker  = data.get('ticker',  'UNKNOWN')
    trades  = data.get('trades',  [])

    try:
        analysis = groq_chat(
            system="You are a senior quantitative analyst. Respond in clear professional prose only. No markdown, no bullet points.",
            prompt=f"""Analyze this trading strategy result professionally.

Ticker: {ticker}
Strategy Return: {metrics.get('total_return')}%  |  Buy & Hold: {metrics.get('market_return')}%
CAGR: {metrics.get('cagr')}%  |  Sharpe: {metrics.get('sharpe_ratio')}  |  Calmar: {metrics.get('calmar_ratio')}
Max Drawdown: {metrics.get('max_drawdown')}%  |  Volatility: {metrics.get('volatility')}%
Win Rate: {metrics.get('win_rate')}%  |  Total Trades: {metrics.get('n_trades')}
Avg Trade P&L: ${metrics.get('avg_trade_pnl')}  |  Final Value: ${metrics.get('final_value',0):,}

Recent trades: {json.dumps(trades[-5:])}

Write exactly 4 professional paragraphs:
1. Alpha generation vs benchmark
2. Risk-adjusted quality (Sharpe, Calmar)
3. Signal quality and win rate
4. Key weakness and one concrete improvement""",
            max_tokens=800, temperature=0.3
        )
    except Exception as e:
        analysis = f"AI analysis unavailable: {str(e)}"

    return cors(jsonify({"analysis": analysis}))


# ── OPTIMIZE ──────────────────────────────────────────────────────────────────
@app.route('/api/optimize', methods=['POST', 'OPTIONS'])
def optimize():
    data    = request.get_json(silent=True) or {}
    metrics = data.get('metrics', {})
    params  = data.get('params',  {})
    ticker  = data.get('ticker',  'UNKNOWN')

    short_ma = params.get('short_ma', 20)
    long_ma  = params.get('long_ma',  50)

    try:
        content = groq_chat(
            system="You are a quant analyst. Return ONLY valid JSON. No markdown, no code blocks.",
            prompt=f"""Suggest optimized MA parameters for this strategy.

Ticker: {ticker} | short_ma: {short_ma} | long_ma: {long_ma}
Sharpe: {metrics.get('sharpe_ratio')} | Return: {metrics.get('total_return')}%
Max DD: {metrics.get('max_drawdown')}% | Win Rate: {metrics.get('win_rate')}%

Return ONLY this JSON:
{{
  "suggestions": [
    {{"param":"Short MA","current":"{short_ma}","suggested":"15","reason":"specific reason"}},
    {{"param":"Long MA","current":"{long_ma}","suggested":"60","reason":"specific reason"}}
  ],
  "summary": "one sentence recommendation"
}}""",
            max_tokens=400, temperature=0.2
        )
        content = content.strip().replace("```json","").replace("```","").strip()
        result  = json.loads(content)
    except json.JSONDecodeError:
        result = {
            "suggestions": [
                {"param":"Short MA","current":str(short_ma),"suggested":str(max(5,short_ma-5)),"reason":"Reduce for more responsive signals."},
                {"param":"Long MA","current":str(long_ma),"suggested":str(long_ma+10),"reason":"Increase for stronger trend confirmation."}
            ],
            "summary": "Consider tightening short MA and widening long MA for better signal quality."
        }
    except Exception as e:
        result = {"suggestions":[],"summary":f"Optimization failed: {str(e)}"}

    return cors(jsonify({"success": True, **result}))


# ── PARSE STRATEGY ────────────────────────────────────────────────────────────
@app.route('/api/parse-strategy', methods=['POST', 'OPTIONS'])
def parse_strategy():
    data = request.get_json(silent=True) or {}
    text = data.get('strategy', '')

    try:
        content = groq_chat(
            system="Return ONLY valid JSON. No markdown, no code blocks.",
            prompt=f"""Convert this trading strategy to JSON rules.

Strategy: {text}

Return ONLY this JSON:
{{"buy":"ma_crossover_up","sell":"ma_crossover_down","risk_management":"description","timeframe":"description","reasoning":"brief explanation"}}

Valid buy:  ma_crossover_up, rsi_oversold, macd_crossover_up, bb_lower_touch, ma_and_rsi
Valid sell: ma_crossover_down, rsi_overbought, macd_crossover_down, bb_upper_touch, ma_or_rsi""",
            max_tokens=300, temperature=0.1
        )
        content = content.strip().replace("```json","").replace("```","").strip()
        parsed  = json.loads(content)
        return cors(jsonify({"success": True, "strategy": parsed}))
    except Exception as e:
        return cors(jsonify({"success": False, "error": str(e)})), 500


# ── LOCAL DEV ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n QuantEdge API — http://localhost:5001\n")
    app.run(debug=True, host='0.0.0.0', port=5001)