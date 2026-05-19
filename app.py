from flask import Flask, request, jsonify, send_from_directory, make_response
import sys
import os
import json
import urllib.request
import urllib.error

# =========================
# LOAD ENV
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
# PYTHON PATH
# =========================
sys.path.insert(0, 'src')

from backtest import run_full_backtest

# =========================
# APP
# =========================
app = Flask(__name__, static_folder='public')

# =========================
# MODEL
# =========================
LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# =========================
# CORS
# =========================
def _cors(response):

    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'

    return response


# =========================
# HOME
# =========================
@app.route('/')
def home():
    return send_from_directory('public', 'index.html')


# =========================
# STATUS
# =========================
@app.route('/api/status')
def status():

    key = os.environ.get('GROQ_API_KEY', '')

    return jsonify({
        "status": "running",
        "server": "QuantEdge",
        "groq_key_loaded": bool(key),
        "model": LLM_MODEL
    })


# =========================
# BACKTEST
# =========================
@app.route('/api/backtest', methods=['POST', 'OPTIONS'])
def backtest():

    if request.method == 'OPTIONS':
        return _cors(make_response())

    try:

        data = request.get_json() or {}

        result = run_full_backtest(
            ticker=data.get('ticker', 'AAPL'),
            start=data.get('start', '2020-01-01'),
            capital=float(data.get('capital', 100000)),
            params={
                'short_ma': int(data.get('short_ma', 20)),
                'long_ma': int(data.get('long_ma', 50)),
            }
        )

        return _cors(jsonify(result))

    except Exception as e:

        print("BACKTEST ERROR:", str(e))

        return _cors(jsonify({
            "error": str(e)
        })), 500


# =========================
# GROQ CALL
# =========================
def groq_chat(prompt, max_tokens=500, temperature=0.2):

    key = os.environ.get('GROQ_API_KEY', '')

    if not key:
        raise Exception("Missing GROQ_API_KEY in .env")

    payload = json.dumps({

        "model": LLM_MODEL,

        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],

        "temperature": temperature,
        "max_tokens": max_tokens

    }).encode("utf-8")

    req = urllib.request.Request(

        "https://api.groq.com/openai/v1/chat/completions",

        data=payload,

        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        },

        method="POST"
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

        data = request.get_json() or {}

        metrics = data.get('metrics', {})
        ticker = data.get('ticker', 'UNKNOWN')

        prompt = f"""
Analyze this trading strategy professionally.

Ticker: {ticker}

Return: {metrics.get('total_return')}
Sharpe Ratio: {metrics.get('sharpe_ratio')}
Max Drawdown: {metrics.get('max_drawdown')}
Win Rate: {metrics.get('win_rate')}

Give one concise professional paragraph.
"""

        analysis = groq_chat(
            prompt=prompt,
            max_tokens=300,
            temperature=0.2
        )

        return _cors(jsonify({
            "analysis": analysis
        }))

    except Exception as e:

        print("ANALYZE ERROR:", str(e))

        return _cors(jsonify({
            "analysis": f"AI analysis unavailable: {str(e)}"
        }))


# =========================
# PARSE STRATEGY
# =========================
@app.route('/api/parse-strategy', methods=['POST', 'OPTIONS'])
def parse_strategy():

    if request.method == 'OPTIONS':
        return _cors(make_response())

    try:

        data = request.get_json() or {}

        strategy_text = data.get('strategy', '')

        prompt = f"""
Convert this trading strategy into structured JSON.

Strategy:
{strategy_text}

Return ONLY valid JSON.

Format:

{{
    "buy": "",
    "sell": "",
    "risk_management": "",
    "timeframe": ""
}}
"""

        content = groq_chat(
            prompt=prompt,
            max_tokens=300,
            temperature=0.1
        )

        # CLEAN MARKDOWN
        content = content.strip()

        if content.startswith("```"):

            content = content.replace("```json", "")
            content = content.replace("```", "")
            content = content.strip()

        try:

            parsed = json.loads(content)

        except Exception:

            parsed = {
                "raw_response": content
            }

        return _cors(jsonify({
            "success": True,
            "strategy": parsed
        }))

    except Exception as e:

        print("PARSE ERROR:", str(e))

        return _cors(jsonify({
            "success": False,
            "error": str(e)
        })), 500


# =========================
# OPTIMIZE
# =========================
@app.route('/api/optimize', methods=['POST', 'OPTIONS'])
def optimize():

    if request.method == 'OPTIONS':
        return _cors(make_response())

    try:

        data = request.get_json() or {}

        metrics = data.get('metrics', {})
        ticker = data.get('ticker', 'UNKNOWN')

        prompt = f"""
Suggest optimized moving average parameters.

Ticker: {ticker}

Sharpe Ratio: {metrics.get('sharpe_ratio')}
Return: {metrics.get('total_return')}
Drawdown: {metrics.get('max_drawdown')}

Return ONLY JSON:

{{
    "short_ma": 10,
    "long_ma": 50,
    "reasoning": ""
}}
"""

        content = groq_chat(
            prompt=prompt,
            max_tokens=300,
            temperature=0.2
        )

        content = content.strip()

        if content.startswith("```"):

            content = content.replace("```json", "")
            content = content.replace("```", "")
            content = content.strip()

        try:

            parsed = json.loads(content)

        except Exception:

            parsed = {
                "short_ma": 20,
                "long_ma": 50,
                "reasoning": content
            }

        return _cors(jsonify({
            "success": True,
            "suggestion": parsed
        }))

    except Exception as e:

        print("OPTIMIZE ERROR:", str(e))

        return _cors(jsonify({
            "success": False,
            "error": str(e)
        })), 500


# =========================
# MAIN
# =========================
if __name__ == '__main__':

    key = os.environ.get('GROQ_API_KEY', '')

    print("\n QuantEdge v2.0")
    print(f" GROQ_API_KEY: {'LOADED' if key else 'NOT FOUND'}")
    print(f" MODEL: {LLM_MODEL}")
    print(" OPEN: http://localhost:5000\n")

    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )