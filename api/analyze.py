"""
Vercel Serverless Function — /api/analyze
AI-powered backtest analysis using Groq (free tier)
"""
import os, json, urllib.request, urllib.error

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'
MODEL        = 'meta-llama/llama-4-scout-17b-16e-instruct'

HEADERS = {
    'Access-Control-Allow-Origin':  '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Content-Type': 'application/json',
}


def call_groq(metrics: dict, ticker: str, trades: list) -> str:
    if not GROQ_API_KEY:
        return (
            f"AI analysis is disabled — GROQ_API_KEY not set. "
            f"The {ticker} strategy returned {metrics.get('total_return', 'N/A')}% "
            f"with a Sharpe ratio of {metrics.get('sharpe_ratio', 'N/A')} and "
            f"{metrics.get('max_drawdown', 'N/A')}% max drawdown. "
            "To enable AI analysis, add GROQ_API_KEY in your Vercel environment variables. "
            "Get a free key at https://console.groq.com"
        )

    prompt = f"""You are a senior quantitative researcher at a hedge fund. 
Analyze this backtesting result. Write exactly 4 professional paragraphs.
No markdown, no bullet points, no headers.

TICKER: {ticker}

METRICS:
- Strategy Return: {metrics.get('total_return')}%
- Buy & Hold Return: {metrics.get('market_return')}%
- CAGR: {metrics.get('cagr')}%
- Sharpe Ratio: {metrics.get('sharpe_ratio')}
- Max Drawdown: {metrics.get('max_drawdown')}%
- Volatility: {metrics.get('volatility')}%
- Win Rate: {metrics.get('win_rate')}%
- Total Trades: {metrics.get('n_trades')}
- Calmar Ratio: {metrics.get('calmar_ratio')}
- Avg Trade P&L: ${metrics.get('avg_trade_pnl')}
- Final Portfolio Value: ${metrics.get('final_value', 0):,}

RECENT TRADES: {json.dumps(trades[-5:], indent=2)}

Cover in 4 paragraphs:
1. Alpha generation vs benchmark
2. Risk-adjusted quality (Sharpe, Calmar)
3. Signal quality and win rate analysis
4. Key weakness and one concrete improvement"""

    payload = json.dumps({
        'model':       MODEL,
        'max_tokens':  800,
        'temperature': 0.3,
        'messages': [
            {'role': 'system', 'content': 'You are a senior quantitative analyst. Respond in clear professional prose only.'},
            {'role': 'user',   'content': prompt}
        ],
    }).encode('utf-8')

    req = urllib.request.Request(
        GROQ_API_URL,
        data=payload,
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {GROQ_API_KEY}'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=35) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    return data['choices'][0]['message']['content']


def handler(request, response=None):
    if request.method == 'OPTIONS':
        return {'statusCode': 200, 'headers': HEADERS, 'body': ''}

    try:
        body    = json.loads(request.body or '{}')
        metrics = body.get('metrics', {})
        ticker  = body.get('ticker',  'UNKNOWN')
        trades  = body.get('trades',  [])

        if not metrics:
            return {'statusCode': 400, 'headers': HEADERS,
                    'body': json.dumps({'error': 'No metrics provided'})}

        analysis = call_groq(metrics, ticker, trades)
        return {'statusCode': 200, 'headers': HEADERS,
                'body': json.dumps({'analysis': analysis})}

    except urllib.error.HTTPError as e:
        err = e.read().decode('utf-8')
        return {'statusCode': e.code, 'headers': HEADERS,
                'body': json.dumps({'analysis': f'Groq API error {e.code}: {err}'})}
    except Exception as e:
        return {'statusCode': 500, 'headers': HEADERS,
                'body': json.dumps({'analysis': f'AI analysis failed: {str(e)}'})}
