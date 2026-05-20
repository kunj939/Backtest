#!/usr/bin/env python
import sys
import json
sys.path.insert(0, '.')

# Simulate what the Flask app does
from backtest_engine import run_full_backtest

# Test with the exact parameters that the frontend sends
data = {
    'ticker': 'AAPL',
    'start': '2020-01-01',
    'capital': 100000,
    'strategy': 'ma_crossover',
    'short_ma': 20,
    'long_ma': 50,
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bb_period': 20,
    'bb_std': 2.0,
}

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

try:
    result = run_full_backtest(
        ticker   = data['ticker'].upper().strip(),
        start    = data.get('start', '2020-01-01'),
        capital  = float(data.get('capital', 100000)),
        strategy = data.get('strategy', 'ma_crossover'),
        params   = params,
    )
    
    # Check if result has all required fields
    if 'error' in result:
        print(f"API Error: {result['error']}")
    else:
        print(f"✓ Backtest successful")
        print(f"  - Total return: {result['metrics']['total_return']}%")
        print(f"  - Chart size: {len(result.get('chart', ''))} bytes")
        print(f"  - Data points: {result['data_points']}")
        
except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
