"""
Vercel Serverless Function — /api/parse-strategy
Parses natural language strategy into structured JSON rules
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


def handler(request, response=None):
    if request.method == 'OPTIONS':
        return {'statusCode': 200, 'headers': HEADERS, 'body': ''}

    try:
        body          = json.loads(request.body or '{}')
        strategy_text = body.get('strategy', '')

        if not GROQ_API_KEY:
            return {'statusCode': 200, 'headers': HEADERS, 'body': json.dumps({
                'success': False,
                'error': 'GROQ_API_KEY not set. Add it to Vercel environment variables.'
            })}

        prompt = f"""Convert this trading strategy into structured JSON rules.

Strategy: {strategy_text}

Return ONLY valid JSON (no markdown, no backticks):

{{
  "buy":  "ma_crossover_up",
  "sell": "ma_crossover_down",
  "risk_management": "description of risk rules",
  "timeframe": "description of timeframe",
  "reasoning": "brief explanation of interpretation"
}}

Valid buy values:  ma_crossover_up, rsi_oversold, macd_crossover_up, bb_lower_touch, ma_and_rsi
Valid sell values: ma_crossover_down, rsi_overbought, macd_crossover_down, bb_upper_touch, ma_or_rsi

Choose the closest matching values."""

        payload = json.dumps({
            'model':       MODEL,
            'max_tokens':  300,
            'temperature': 0.1,
            'messages': [
                {'role': 'system', 'content': 'Return ONLY valid JSON. No markdown, no code blocks.'},
                {'role': 'user',   'content': prompt}
            ],
        }).encode('utf-8')

        req = urllib.request.Request(
            GROQ_API_URL,
            data=payload,
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {GROQ_API_KEY}'},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        content = data['choices'][0]['message']['content'].strip()
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(content)
        except Exception:
            parsed = {"raw_response": content}

        return {'statusCode': 200, 'headers': HEADERS,
                'body': json.dumps({'success': True, 'strategy': parsed})}

    except Exception as e:
        return {'statusCode': 500, 'headers': HEADERS,
                'body': json.dumps({'success': False, 'error': str(e)})}
