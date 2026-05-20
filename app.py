from flask import Flask, request, jsonify, send_from_directory, make_response
import sys
import os
import json
import urllib.request
import urllib.error

# =========================
# LOAD .env FILE
# =========================
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, val = line.partition('=')
                    os.environ[key.strip()] = val.strip()

load_env()

# =========================
# IMPORT BACKTEST ENGINE
# =========================
sys.path.insert(0, os.path.dirname(__file__))
from backtest_engine import run_full_backtest

# =========================
# FLASK APP
# =========================
app = Flask(__name__, static_folder='public')

LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


# =========================
# CORS HELPER
# =========================
def _cors(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response


# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return send_from_directory('public', 'index.html')


@app.route('/api/status')
def status():
    key = os.environ.get('GROQ_API_KEY', '')
    return _cors(jsonify({
        "status":          "running",
        "server":          "QuantEdge",
        "groq_key_loaded": bool(key),
        "model":           LLM_MODEL
    }))


# =========================
# BACKTEST
# =========================
@app.route('/api/backtest', methods=['POST', 'OPTIONS'])
def backtest():
    if request.method == 'OPTIONS':
        return _cors(make_response())

    try:
        data = request.get_json() or {}

        ticker   = data.get('ticker',   'AAPL').upper().strip()
        start    = data.get('start',    '2020-01-01')
        capital  = float(data.get('capital', 100000))
        strategy = data.get('strategy', 'ma_crossover')

        params = {
            'short_ma':     int(data.get('short_ma',     20)),
            'long_ma':      int(data.get('long_ma',      50)),
            'rsi_period':   int(data.get('rsi_period',   14)),
            'rsi_overbought': int(data.get('rsi_overbought', 70)),
            'rsi_oversold':   int(data.get('rsi_oversold',   30)),
            'macd_fast':    int(data.get('macd_fast',    12)),
            'macd_slow':    int(data.get('macd_slow',    26)),
            'macd_signal':  int(data.get('macd_signal',   9)),
            'bb_period':    int(data.get('bb_period',    20)),
            'bb_std':     float(data.get('bb_std',      2.0)),
        }

        # Custom strategy rules
        if strategy == 'custom' and data.get('custom_rules'):
            params['custom_rules'] = data['custom_rules']

        result = run_full_backtest(
            ticker   = ticker,
            start    = start,
            capital  = capital,
            strategy = strategy,
            params   = params,
        )

        return _cors(jsonify(result))

    except Exception as e:
        print("BACKTEST ERROR:", str(e))
        return _cors(jsonify({"error": str(e)})), 500


# =========================
# GROQ HELPER
# =========================
def groq_chat(prompt, system=None, max_tokens=800, temperature=0.3):
    key = os.environ.get('GROQ_API_KEY', '')
    if not key:
        raise Exception("Missing GROQ_API_KEY. Add it to .env or Vercel environment variables.")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model":       LLM_MODEL,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data    = payload,
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type":  "application/json"
        },
        method  = "POST"
    )

    with urllib.request.urlopen(req, timeout=40) as response:
        result = json.loads(response.read())

    return result["choices"][0]["message"]["content"]


# =========================
# AI ANALYZE
# =========================
@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return _cors(make_response())

    try:
        data    = request.get_json() or {}
        metrics = data.get('metrics', {})
        ticker  = data.get('ticker',  'UNKNOWN')
        trades  = data.get('trades',  [])

        system = (
            "You are a senior quantitative analyst at a hedge fund. "
            "Respond in clear, professional prose. No markdown, no bullet points, no headers."
        )

        prompt = f"""Analyze this trading strategy result professionally.

Ticker: {ticker}
Strategy Return: {metrics.get('total_return')}%
Buy & Hold Return: {metrics.get('market_return')}%
CAGR: {metrics.get('cagr')}%
Sharpe Ratio: {metrics.get('sharpe_ratio')}
Max Drawdown: {metrics.get('max_drawdown')}%
Volatility: {metrics.get('volatility')}%
Win Rate: {metrics.get('win_rate')}%
Total Trades: {metrics.get('n_trades')}
Calmar Ratio: {metrics.get('calmar_ratio')}
Avg Trade P&L: ${metrics.get('avg_trade_pnl')}
Final Value: ${metrics.get('final_value', 0):,}

Recent trades: {json.dumps(trades[-5:], indent=2)}

Write exactly 4 professional paragraphs covering:
1. Alpha vs benchmark
2. Risk-adjusted quality (Sharpe, Calmar)
3. Signal quality and trade frequency
4. Key weakness and one concrete improvement suggestion"""

        analysis = groq_chat(prompt=prompt, system=system, max_tokens=800, temperature=0.3)
        return _cors(jsonify({"analysis": analysis}))

    except Exception as e:
        print("ANALYZE ERROR:", str(e))
        return _cors(jsonify({"analysis": f"AI analysis unavailable: {str(e)}"}))


# =========================
# AI OPTIMIZE
# =========================
@app.route('/api/optimize', methods=['POST', 'OPTIONS'])
def optimize():
    if request.method == 'OPTIONS':
        return _cors(make_response())

    try:
        data    = request.get_json() or {}
        metrics = data.get('metrics', {})
        params  = data.get('params',  {})
        ticker  = data.get('ticker',  'UNKNOWN')

        prompt = f"""Suggest optimized parameters for this trading strategy.

Ticker: {ticker}
Current short_ma: {params.get('short_ma', 20)}
Current long_ma: {params.get('long_ma', 50)}
Sharpe Ratio: {metrics.get('sharpe_ratio')}
Total Return: {metrics.get('total_return')}%
Max Drawdown: {metrics.get('max_drawdown')}%
Win Rate: {metrics.get('win_rate')}%
Calmar Ratio: {metrics.get('calmar_ratio')}

Return ONLY valid JSON (no markdown, no explanation outside JSON):

{{
  "suggestions": [
    {{
      "param": "Short MA",
      "current": "{params.get('short_ma', 20)}",
      "suggested": "15",
      "reason": "explanation"
    }},
    {{
      "param": "Long MA",
      "current": "{params.get('long_ma', 50)}",
      "suggested": "60",
      "reason": "explanation"
    }}
  ],
  "summary": "one sentence overall recommendation"
}}"""

        content = groq_chat(prompt=prompt, max_tokens=500, temperature=0.2)

        # Strip markdown fences if present
        content = content.strip()
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(content)
        except Exception:
            # Fallback — return safe default
            parsed = {
                "suggestions": [
                    {
                        "param":     "Short MA",
                        "current":   str(params.get('short_ma', 20)),
                        "suggested": "15",
                        "reason":    content[:200] if content else "AI response could not be parsed."
                    }
                ],
                "summary": "Could not parse full AI response."
            }

        return _cors(jsonify({"success": True, **parsed}))

    except Exception as e:
        print("OPTIMIZE ERROR:", str(e))
        return _cors(jsonify({"success": False, "error": str(e)})), 500


# =========================
# PARSE STRATEGY (AI Custom)
# =========================
@app.route('/api/parse-strategy', methods=['POST', 'OPTIONS'])
def parse_strategy():
    if request.method == 'OPTIONS':
        return _cors(make_response())

    try:
        data          = request.get_json() or {}
        strategy_text = data.get('strategy', '')

        prompt = f"""Convert this trading strategy description into structured JSON rules.

Strategy: {strategy_text}

Return ONLY valid JSON (no markdown fences):

{{
  "buy":  "ma_crossover_up",
  "sell": "ma_crossover_down",
  "risk_management": "description",
  "timeframe": "description",
  "reasoning": "brief explanation"
}}

Valid buy values:  ma_crossover_up, rsi_oversold, macd_crossover_up, bb_lower_touch, ma_and_rsi
Valid sell values: ma_crossover_down, rsi_overbought, macd_crossover_down, bb_upper_touch, ma_or_rsi"""

        content = groq_chat(prompt=prompt, max_tokens=300, temperature=0.1)
        content = content.strip()
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(content)
        except Exception:
            parsed = {"raw_response": content}

        return _cors(jsonify({"success": True, "strategy": parsed}))

    except Exception as e:
        print("PARSE ERROR:", str(e))
        return _cors(jsonify({"success": False, "error": str(e)})), 500


# =========================
# MAIN
# =========================
if __name__ == '__main__':
    key = os.environ.get('GROQ_API_KEY', '')
    print("\n QuantEdge v2.1")
    print(f" GROQ_API_KEY: {'LOADED ✓' if key else 'NOT FOUND — AI features disabled'}")
    print(f" MODEL: {LLM_MODEL}")
    print(" OPEN: http://localhost:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
