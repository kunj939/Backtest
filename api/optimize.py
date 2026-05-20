"""
Vercel Serverless Function — /api/optimize
AI parameter optimization using Groq
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


def call_groq(metrics: dict, params: dict, ticker: str) -> dict:
    if not GROQ_API_KEY:
        return {
            "suggestions": [
                {
                    "param":     "API Key",
                    "current":   "missing",
                    "suggested": "required",
                    "reason":    "Add GROQ_API_KEY to Vercel env vars to enable AI optimization. Get a free key at https://console.groq.com"
                }
            ],
            "summary": "AI optimization requires a GROQ_API_KEY environment variable."
        }

    short_ma = params.get('short_ma', 20)
    long_ma  = params.get('long_ma',  50)

    prompt = f"""Suggest optimized parameters for this MA crossover trading strategy.

Ticker: {ticker}
Current short_ma: {short_ma}
Current long_ma: {long_ma}
Sharpe Ratio: {metrics.get('sharpe_ratio')}
Total Return: {metrics.get('total_return')}%
Max Drawdown: {metrics.get('max_drawdown')}%
Win Rate: {metrics.get('win_rate')}%
Total Trades: {metrics.get('n_trades')}
Calmar Ratio: {metrics.get('calmar_ratio')}

Return ONLY valid JSON (absolutely no markdown, no backticks, no explanation outside JSON):

{{
  "suggestions": [
    {{
      "param": "Short MA",
      "current": "{short_ma}",
      "suggested": "15",
      "reason": "specific reason based on the metrics"
    }},
    {{
      "param": "Long MA",
      "current": "{long_ma}",
      "suggested": "60",
      "reason": "specific reason based on the metrics"
    }}
  ],
  "summary": "one sentence overall recommendation"
}}"""

    payload = json.dumps({
        'model':       MODEL,
        'max_tokens':  500,
        'temperature': 0.2,
        'messages': [
            {'role': 'system', 'content': 'You are a quant analyst. Return ONLY valid JSON. No markdown, no code blocks, no explanation outside JSON.'},
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

    content = data['choices'][0]['message']['content'].strip()
    # Strip markdown fences if model still adds them
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(content)
    except Exception:
        return {
            "suggestions": [
                {
                    "param":     "Short MA",
                    "current":   str(short_ma),
                    "suggested": str(max(5, short_ma - 5)),
                    "reason":    content[:200] if content else "Could not parse AI response."
                }
            ],
            "summary": "AI response could not be fully parsed."
        }


def handler(request, response=None):
    if request.method == 'OPTIONS':
        return {'statusCode': 200, 'headers': HEADERS, 'body': ''}

    try:
        body    = json.loads(request.body or '{}')
        metrics = body.get('metrics', {})
        params  = body.get('params',  {})
        ticker  = body.get('ticker',  'UNKNOWN')

        result = call_groq(metrics, params, ticker)
        return {'statusCode': 200, 'headers': HEADERS,
                'body': json.dumps({'success': True, **result})}

    except urllib.error.HTTPError as e:
        err = e.read().decode('utf-8')
        return {'statusCode': e.code, 'headers': HEADERS,
                'body': json.dumps({'success': False, 'error': f'Groq API error: {err}'})}
    except Exception as e:
        return {'statusCode': 500, 'headers': HEADERS,
                'body': json.dumps({'success': False, 'error': str(e)})}
