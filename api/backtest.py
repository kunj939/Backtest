"""
Vercel Serverless Function — /api/backtest
Handles POST requests from the frontend to run backtests.
"""

import sys
import os
import json

# Add src to path for Vercel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtest import run_full_backtest


def handler(request, response=None):
    """
    Vercel serverless handler.
    Accepts POST with JSON body: { ticker, start, end, capital, short_ma, long_ma }
    """
    # Handle CORS preflight
    headers = {
        'Access-Control-Allow-Origin':  '*',
        'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Content-Type': 'application/json',
    }
    
    if request.method == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers':    headers,
            'body':       '',
        }
    
    try:
        body = json.loads(request.body or '{}')
        
        ticker   = body.get('ticker',   'AAPL').upper().strip()
        start    = body.get('start',    '2020-01-01')
        end      = body.get('end',      None)
        capital  = float(body.get('capital',  100_000))
        short_ma = int(body.get('short_ma', 20))
        long_ma  = int(body.get('long_ma',  50))
        
        # Validation
        if short_ma >= long_ma:
            return {
                'statusCode': 400,
                'headers':    headers,
                'body':       json.dumps({'error': 'short_ma must be less than long_ma'}),
            }
        if capital < 1000:
            return {
                'statusCode': 400,
                'headers':    headers,
                'body':       json.dumps({'error': 'Minimum capital is $1,000'}),
            }
        
        result = run_full_backtest(
            ticker=ticker,
            start=start,
            end=end,
            short_ma=short_ma,
            long_ma=long_ma,
            capital=capital,
        )
        
        return {
            'statusCode': 200,
            'headers':    headers,
            'body':       json.dumps(result),
        }
    
    except ValueError as e:
        return {
            'statusCode': 400,
            'headers':    headers,
            'body':       json.dumps({'error': str(e)}),
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers':    headers,
            'body':       json.dumps({'error': f'Internal error: {str(e)}'}),
        }
