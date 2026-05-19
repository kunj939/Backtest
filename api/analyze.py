"""
Vercel Serverless Function — /api/analyze
LLM Agent endpoint: sends backtest results to Groq (FREE) for quant analysis.
Model: llama-3.3-70b-versatile (Groq free tier)
Get free API key at: https://console.groq.com
"""

import json
import os
import urllib.request
import urllib.error


GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'
MODEL        = 'llama-3.3-70b-versatile'


def call_groq(metrics: dict, ticker: str, trades: list) -> str:
    """
    Calls Groq API (free) with Llama 3.3 70B to generate
    a quant researcher's interpretation of backtest results.
    """
    prompt = f"""You are a senior quantitative researcher at a hedge fund. 
Analyze this backtesting result and provide sharp, data-driven insights in 4 paragraphs.
Be concise, professional, and specific. Write like a Bloomberg intelligence brief.

STRATEGY: Moving Average Crossover (MA-20 vs MA-50)
TICKER: {ticker}

PERFORMANCE METRICS:
- Strategy Total Return: {metrics['total_return']}%
- Buy & Hold Return: {metrics['market_return']}%
- CAGR: {metrics['cagr']}%
- Sharpe Ratio: {metrics['sharpe_ratio']}
- Max Drawdown: {metrics['max_drawdown']}%
- Annualized Volatility: {metrics['volatility']}%
- Win Rate: {metrics['win_rate']}%
- Total Trades: {metrics['n_trades']}
- Calmar Ratio: {metrics['calmar_ratio']}
- Avg Trade P&L: ${metrics['avg_trade_pnl']}
- Final Portfolio Value: ${metrics['final_value']:,}

RECENT TRADES (last {len(trades)}):
{json.dumps(trades, indent=2)}

Your analysis must cover:
1. Alpha generation vs benchmark (did the strategy beat buy-and-hold and by how much?)
2. Risk-adjusted quality (Sharpe, Calmar interpretation — is this investment-grade?)
3. Signal quality (win rate, trade frequency, drawdown resilience)
4. Key weakness + one concrete improvement suggestion (RSI filter, volatility regime, ML layer, etc.)

Format: exactly 4 paragraphs. No bullet points. No headers. Professional quant voice."""

    payload = json.dumps({
        'model':       MODEL,
        'max_tokens':  800,
        'temperature': 0.4,
        'messages': [
            {
                'role':    'system',
                'content': 'You are a senior quantitative analyst. Respond in clear, professional prose only. No markdown, no bullet points, no headers.'
            },
            {
                'role':    'user',
                'content': prompt
            }
        ],
    }).encode('utf-8')

    req = urllib.request.Request(
        GROQ_API_URL,
        data=payload,
        headers={
            'Content-Type':  'application/json',
            'Authorization': f'Bearer {GROQ_API_KEY}',
        },
        method='POST',
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        return data['choices'][0]['message']['content']


def handler(request, response=None):
    headers = {
        'Access-Control-Allow-Origin':  '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Content-Type':                 'application/json',
    }

    if request.method == 'OPTIONS':
        return {'statusCode': 200, 'headers': headers, 'body': ''}

    try:
        body    = json.loads(request.body or '{}')
        metrics = body.get('metrics', {})
        ticker  = body.get('ticker', 'UNKNOWN')
        trades  = body.get('trades', [])

        if not metrics:
            return {
                'statusCode': 400,
                'headers':    headers,
                'body':       json.dumps({'error': 'No metrics provided'}),
            }

        if not GROQ_API_KEY:
            analysis = (
                f"The MA crossover strategy on {ticker} returned {metrics.get('total_return', 'N/A')}% "
                f"with a Sharpe ratio of {metrics.get('sharpe_ratio', 'N/A')}. "
                "To enable AI-powered analysis, add GROQ_API_KEY in Vercel environment variables. "
                "Get a free key at https://console.groq.com"
            )
        else:
            analysis = call_groq(metrics, ticker, trades)

        return {
            'statusCode': 200,
            'headers':    headers,
            'body':       json.dumps({'analysis': analysis}),
        }

    except urllib.error.HTTPError as e:
        err_body = e.read().decode('utf-8')
        return {
            'statusCode': e.code,
            'headers':    headers,
            'body':       json.dumps({'error': f'Groq API error {e.code}: {err_body}'}),
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers':    headers,
            'body':       json.dumps({'error': str(e)}),
        }
