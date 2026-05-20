#!/usr/bin/env python
import json
import sys
sys.path.insert(0, '.')
from backtest_engine import run_full_backtest

try:
    result = run_full_backtest(
        ticker='AAPL',
        start='2020-01-01',
        strategy='ma_crossover',
        params={'short_ma': 20, 'long_ma': 50},
        capital=100000
    )
    print('SUCCESS! Backtest ran without errors')
    metrics = result['metrics']
    print(f"Total return: {metrics['total_return']}%")
    print(f"Sharpe ratio: {metrics['sharpe_ratio']}")
except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
