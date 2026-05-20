"""
Vercel Serverless Function — /api/backtest
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backtest_engine import run_full_backtest

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
        data = json.loads(request.body or '{}')

        ticker   = data.get('ticker',   'AAPL').upper().strip()
        start    = data.get('start',    '2020-01-01')
        capital  = float(data.get('capital', 100_000))
        strategy = data.get('strategy', 'ma_crossover')

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

        # Validations
        if params['short_ma'] >= params['long_ma']:
            return {'statusCode': 400, 'headers': HEADERS,
                    'body': json.dumps({'error': 'short_ma must be less than long_ma'})}
        if capital < 1000:
            return {'statusCode': 400, 'headers': HEADERS,
                    'body': json.dumps({'error': 'Minimum capital is $1,000'})}

        result = run_full_backtest(
            ticker=ticker, start=start, capital=capital,
            strategy=strategy, params=params
        )
        return {'statusCode': 200, 'headers': HEADERS, 'body': json.dumps(result)}

    except ValueError as e:
        return {'statusCode': 400, 'headers': HEADERS, 'body': json.dumps({'error': str(e)})}
    except Exception as e:
        return {'statusCode': 500, 'headers': HEADERS, 'body': json.dumps({'error': f'Server error: {str(e)}'})}
