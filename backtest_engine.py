"""
QuantEdge Backtest Engine v2.4
- Fixed: maximum recursion depth exceeded on Render
- Fixed: pandas 2.x/3.x CoW inplace deprecation
- Fixed: numpy log on zero/NaN causing chain errors
Data: Yahoo Finance v8 API (direct) → Stooq CSV fallback
"""
import sys
import os
import pandas as pd
import numpy as np
import urllib.request
import json
import io as _io
import gzip
import warnings
import io
import base64
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime

# Prevent recursion issues on Render
sys.setrecursionlimit(5000)
warnings.filterwarnings('ignore')
try:
    pd.options.mode.copy_on_write = False
except Exception:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCH
# ══════════════════════════════════════════════════════════════════════════════

def _dt_to_ts(date_str):
    return int(datetime.strptime(date_str, '%Y-%m-%d').timestamp())


def _parse_yahoo_response(data):
    result = data.get('chart', {}).get('result')
    if not result:
        return None
    r0 = result[0]
    timestamps = r0.get('timestamp', [])
    if not timestamps:
        return None
    q = r0['indicators']['quote'][0]
    adj_close = (
        r0['indicators'].get('adjclose', [{}])[0].get('adjclose')
        or q['close']
    )
    df = pd.DataFrame({
        'Open':   q['open'],
        'High':   q['high'],
        'Low':    q['low'],
        'Close':  adj_close,
        'Volume': q['volume'],
    }, index=pd.to_datetime(timestamps, unit='s').normalize())
    df.index.name = 'Date'
    df = df.dropna(subset=['Close'])
    df = df.sort_index()
    return df if len(df) >= 60 else None


def _yahoo_request(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                enc = r.headers.get('Content-Encoding', '')
                if enc == 'gzip':
                    import gzip as gz
                    raw = gz.decompress(raw)
            return json.loads(raw.decode('utf-8'))
        except Exception as e:
            if attempt < 2:
                time.sleep(1 + attempt)
            else:
                raise


def _fetch_yahoo(ticker, start, end, host='query1'):
    period1 = _dt_to_ts(start)
    period2 = _dt_to_ts(end)
    url = (
        f"https://{host}.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={period1}&period2={period2}&interval=1d"
        f"&includeAdjustedClose=true"
    )
    data = _yahoo_request(url)
    return _parse_yahoo_response(data)


def _fetch_stooq(ticker, start, end):
    sym = ticker.lower()
    if '.' not in sym:
        sym += '.us'
    d1  = start.replace('-', '')
    d2  = end.replace('-', '')
    url = f"https://stooq.com/q/d/l/?s={sym}&d1={d1}&d2={d2}&i=d"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv",
        "Referer": "https://stooq.com/",
    }
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as r:
                content = r.read().decode('utf-8')
            if 'No data' in content or len(content) < 100:
                return None
            df = pd.read_csv(_io.StringIO(content), parse_dates=['Date'], index_col='Date')
            df.columns = [c.strip().title() for c in df.columns]
            df = df.sort_index()
            df = df.dropna(subset=['Close'])
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df if len(df) >= 60 else None
        except Exception:
            if attempt < 1:
                time.sleep(1)
            else:
                return None


def fetch_data(ticker, start, end=None):
    ticker = ticker.strip().upper().split()[0]
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    errors = []

    for host in ['query1', 'query2']:
        try:
            df = _fetch_yahoo(ticker, start, end, host=host)
            if df is not None:
                print(f"[fetch] ✓ Yahoo {host} ({len(df)} rows)")
                return _clean_df(df)
        except Exception as e:
            errors.append(f"Yahoo {host}: {e}")

    try:
        df = _fetch_stooq(ticker, start, end)
        if df is not None:
            print(f"[fetch] ✓ Stooq ({len(df)} rows)")
            return _clean_df(df)
    except Exception as e:
        errors.append(f"Stooq: {e}")

    raise ValueError(
        f"Could not fetch data for '{ticker}'. "
        f"Tried: {' | '.join(errors)}. "
        f"Check the ticker symbol (e.g. AAPL, TSLA, INFY.NS) and internet connection."
    )


def _clean_df(df):
    cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
    df = df[cols].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    if len(df) < 60:
        raise ValueError(f"Not enough data ({len(df)} rows). Use a longer date range.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS  — all use .copy() + assignment, no inplace
# ══════════════════════════════════════════════════════════════════════════════

def add_indicators(df, p):
    df = df.copy()

    short = p.get('short_ma', 20)
    long_ = p.get('long_ma',  50)
    df['MA_Short'] = df['Close'].rolling(short, min_periods=1).mean()
    df['MA_Long']  = df['Close'].rolling(long_,  min_periods=1).mean()

    rp    = p.get('rsi_period', 14)
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(rp).mean()
    loss  = (-delta.clip(upper=0)).rolling(rp).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    ema_f = df['Close'].ewm(span=p.get('macd_fast', 12),   adjust=False).mean()
    ema_s = df['Close'].ewm(span=p.get('macd_slow', 26),   adjust=False).mean()
    df['MACD']        = ema_f - ema_s
    df['MACD_Signal'] = df['MACD'].ewm(span=p.get('macd_signal', 9), adjust=False).mean()

    bp = p.get('bb_period', 20)
    bs = p.get('bb_std',    2.0)
    df['BB_Mid']   = df['Close'].rolling(bp).mean()
    df['BB_Std']   = df['Close'].rolling(bp).std()
    df['BB_Upper'] = df['BB_Mid'] + bs * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - bs * df['BB_Std']

    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low']  - df['Close'].shift()).abs()
    df['ATR'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def _add_signal(df):
    df['Signal'] = df['Position'].diff()
    return df


def strat_ma(df, p):
    df = df.copy()
    lm = p.get('long_ma', 50)
    pos = np.where(df['MA_Short'] > df['MA_Long'], 1, 0)
    df['Position'] = pos
    df = _add_signal(df)
    df.loc[df.index[:lm], ['Signal', 'Position']] = 0
    return df


def strat_rsi(df, p):
    df   = df.copy()
    pos  = [0] * len(df)
    cur  = 0
    rsi  = df['RSI'].values
    ob   = p.get('rsi_overbought', 70)
    os_  = p.get('rsi_oversold',   30)
    for i, v in enumerate(rsi):
        if not np.isnan(v):
            if v < os_:  cur = 1
            elif v > ob: cur = 0
        pos[i] = cur
    df['Position'] = pos
    df = _add_signal(df)
    return df


def strat_macd(df, p):
    df = df.copy()
    df['Position'] = np.where(df['MACD'] > df['MACD_Signal'], 1, 0)
    df = _add_signal(df)
    df.loc[df.index[:30], ['Signal', 'Position']] = 0
    return df


def strat_bb(df, p):
    df   = df.copy()
    pos  = [0] * len(df)
    cur  = 0
    close   = df['Close'].values
    bb_up   = df['BB_Upper'].values
    bb_lo   = df['BB_Lower'].values
    for i in range(len(df)):
        if not np.isnan(bb_up[i]):
            if   close[i] <= bb_lo[i]: cur = 1
            elif close[i] >= bb_up[i]: cur = 0
        pos[i] = cur
    df['Position'] = pos
    df = _add_signal(df)
    return df


def strat_combined(df, p):
    df  = df.copy()
    lm  = p.get('long_ma', 50)
    ob  = p.get('rsi_overbought', 70)
    pos = [0] * len(df)
    cur = 0
    ma_s  = df['MA_Short'].values
    ma_l  = df['MA_Long'].values
    rsi   = df['RSI'].values
    for i in range(len(df)):
        if i < lm:
            pos[i] = 0
            continue
        ma_ok  = ma_s[i] > ma_l[i]
        rsi_ok = not np.isnan(rsi[i]) and rsi[i] < ob
        if ma_ok and rsi_ok:  cur = 1
        elif not ma_ok:       cur = 0
        pos[i] = cur
    df['Position'] = pos
    df = _add_signal(df)
    return df


def strat_custom(df, p):
    rules     = p.get('custom_rules', {})
    buy_rule  = rules.get('buy',  'ma_crossover_up')
    sell_rule = rules.get('sell', 'ma_crossover_down')
    df   = df.copy()
    pos  = [0] * len(df)
    cur  = 0
    for i in range(len(df)):
        row = df.iloc[i]
        buy = sell = False
        ma_s  = row['MA_Short']
        ma_l  = row['MA_Long']
        rsi_v = row['RSI']
        macd_ = row['MACD']
        macs_ = row['MACD_Signal']
        bbu   = row['BB_Upper']
        bbl   = row['BB_Lower']
        cl    = row['Close']
        os_   = p.get('rsi_oversold',   30)
        ob_   = p.get('rsi_overbought', 70)

        if   buy_rule == 'ma_crossover_up':     buy = ma_s > ma_l
        elif buy_rule == 'rsi_oversold':        buy = not np.isnan(rsi_v) and rsi_v < os_
        elif buy_rule == 'macd_crossover_up':   buy = not np.isnan(macd_) and macd_ > macs_
        elif buy_rule == 'bb_lower_touch':      buy = not np.isnan(bbl)   and cl <= bbl
        elif buy_rule == 'ma_and_rsi':          buy = ma_s > ma_l and not np.isnan(rsi_v) and rsi_v < 60

        if   sell_rule == 'ma_crossover_down':     sell = ma_s < ma_l
        elif sell_rule == 'rsi_overbought':        sell = not np.isnan(rsi_v) and rsi_v > ob_
        elif sell_rule == 'macd_crossover_down':   sell = not np.isnan(macd_) and macd_ < macs_
        elif sell_rule == 'bb_upper_touch':        sell = not np.isnan(bbu)   and cl >= bbu
        elif sell_rule == 'ma_or_rsi':             sell = ma_s < ma_l or (not np.isnan(rsi_v) and rsi_v > 70)

        if buy:  cur = 1
        if sell: cur = 0
        pos[i] = cur

    df['Position'] = pos
    df = _add_signal(df)
    return df


STRATEGY_MAP = {
    'ma_crossover': strat_ma,
    'rsi':          strat_rsi,
    'macd':         strat_macd,
    'bollinger':    strat_bb,
    'combined':     strat_combined,
    'custom':       strat_custom,
}
STRATEGY_LABELS = {
    'ma_crossover': 'MA Crossover',
    'rsi':          'RSI Mean-Reversion',
    'macd':         'MACD Crossover',
    'bollinger':    'Bollinger Bands',
    'combined':     'MA + RSI Combined',
    'custom':       'Custom Strategy',
}


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST + METRICS  — no inplace operations anywhere
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(df, capital):
    df = df.copy()

    close    = df['Close'].values.astype(float)
    position = df['Position'].values.astype(float)

    # Log returns using numpy directly — avoids pandas CoW chain
    mkt_ret  = np.zeros(len(close))
    mkt_ret[1:] = np.log(np.where(close[:-1] > 0, close[1:] / close[:-1], 1.0))

    strat_ret = mkt_ret * np.roll(position, 1)
    strat_ret[0] = 0.0

    mkt_cum   = capital * np.exp(np.nancumsum(mkt_ret))
    strat_cum = capital * np.exp(np.nancumsum(strat_ret))

    df['Market_Return']       = mkt_ret
    df['Strategy_Return']     = strat_ret
    df['Market_Cumulative']   = mkt_cum
    df['Strategy_Cumulative'] = strat_cum

    # Fill any NaN without inplace
    df = df.ffill()
    df = df.fillna(0)
    return df


def compute_metrics(df, capital, rfr=0.05):
    sr = df['Strategy_Return'].replace([np.inf, -np.inf], 0).dropna()
    fv = float(df['Strategy_Cumulative'].iloc[-1])
    mf = float(df['Market_Cumulative'].iloc[-1])
    ny = max(len(df) / 252, 0.01)

    tr   = (fv - capital) / capital * 100
    mr   = (mf - capital) / capital * 100
    cagr = ((fv / capital) ** (1 / ny) - 1) * 100 if fv > 0 else -100.0

    ex    = sr - (rfr / 252)
    std   = float(ex.std())
    sharpe = round(
        float(np.clip((ex.mean() / std) * np.sqrt(252), -50, 50)), 3
    ) if std > 1e-10 else 0.0

    vol   = float(sr.std() * np.sqrt(252) * 100)
    cum   = df['Strategy_Cumulative'].values
    peak  = np.maximum.accumulate(cum)
    dd    = np.where(peak > 0, (cum - peak) / peak * 100, 0)
    maxdd = float(dd.min())

    bis = df[df['Signal'] == 1].index
    sis = df[df['Signal'] == -1].index
    trades, wins = [], []
    for bd in bis:
        fut = sis[sis > bd]
        if len(fut):
            sd  = fut[0]
            pnl = float(df.loc[sd, 'Close']) - float(df.loc[bd, 'Close'])
            trades.append(pnl)
            wins.append(1 if pnl > 0 else 0)

    wr     = sum(wins) / len(wins) * 100 if wins else 0
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0

    return {
        "total_return":    round(tr, 2),
        "market_return":   round(mr, 2),
        "cagr":            round(cagr, 2),
        "sharpe_ratio":    sharpe,
        "max_drawdown":    round(maxdd, 2),
        "volatility":      round(vol, 2),
        "win_rate":        round(wr, 2),
        "n_trades":        len(trades),
        "avg_trade_pnl":   round(float(np.mean(trades)) if trades else 0, 2),
        "calmar_ratio":    round(calmar, 3),
        "final_value":     round(fv, 2),
        "initial_capital": capital,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CHART
# ══════════════════════════════════════════════════════════════════════════════

def generate_chart(df, ticker, metrics, label):
    BG     = '#0d1117'; PANEL  = '#161b22'; GREEN  = '#39d353'
    RED    = '#f85149'; BLUE   = '#58a6ff'; YELLOW = '#e3b341'
    GRAY   = '#8b949e'; WHITE  = '#f0f6fc'; ACCENT = '#1f6feb'

    plt.rcParams.update({
        'font.family': 'monospace', 'text.color': WHITE,
        'axes.labelcolor': GRAY, 'xtick.color': GRAY, 'ytick.color': GRAY,
        'figure.facecolor': BG, 'axes.facecolor': PANEL,
        'axes.edgecolor': '#30363d', 'grid.color': '#21262d', 'grid.linewidth': 0.5,
    })

    fig = plt.figure(figsize=(18, 15), facecolor=BG)
    gs  = GridSpec(3, 1, figure=fig, hspace=0.08, height_ratios=[3, 2, 1.4])
    ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    ax1.plot(df.index, df['Close'],    color=WHITE,  lw=2.0, alpha=0.95, label='Close')
    ax1.plot(df.index, df['MA_Short'], color=BLUE,   lw=2.0, ls='--',   label='MA-Short')
    ax1.plot(df.index, df['MA_Long'],  color=YELLOW, lw=2.0, ls='--',   label='MA-Long')
    buys  = df[df['Signal'] == 1]
    sells = df[df['Signal'] == -1]
    ax1.scatter(buys.index,  buys['Close'],  marker='^', color=GREEN, s=140, zorder=6, label=f'Buy ({len(buys)})')
    ax1.scatter(sells.index, sells['Close'], marker='v', color=RED,   s=140, zorder=6, label=f'Sell ({len(sells)})')
    ax1.fill_between(df.index, df['Close'].min(), df['Close'].max(),
                     where=(df['Position'] == 1), alpha=0.05, color=GREEN)
    ax1.set_title(f'  {ticker} | {label}  ·  QuantEdge v2.4',
                  color=WHITE, fontsize=13, fontweight='bold', loc='left', pad=12)
    ax1.legend(loc='upper left', framealpha=0.3, fontsize=8, ncol=6)
    ax1.set_ylabel('Price (USD)', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelbottom=False)
    ann = (f"Return: {metrics['total_return']:+.1f}%  |  Sharpe: {metrics['sharpe_ratio']:.2f}"
           f"  |  Max DD: {metrics['max_drawdown']:.1f}%  |  Win: {metrics['win_rate']:.1f}%"
           f"  |  Trades: {metrics['n_trades']}")
    ax1.annotate(ann, xy=(0.01, 0.02), xycoords='axes fraction', fontsize=8.5, color=GRAY,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d1117', alpha=0.7))

    ax2.plot(df.index, df['Strategy_Cumulative'], color=GREEN,  lw=2.4,   label='Strategy')
    ax2.plot(df.index, df['Market_Cumulative'],   color=ACCENT, lw=2.0, label='Buy & Hold', alpha=0.85)
    ax2.axhline(y=metrics['initial_capital'], color=GRAY, ls=':', lw=0.8, alpha=0.5)
    ax2.fill_between(df.index, df['Strategy_Cumulative'], df['Market_Cumulative'],
        where=(df['Strategy_Cumulative'] >= df['Market_Cumulative']),
        alpha=0.12, color=GREEN, interpolate=True)
    ax2.fill_between(df.index, df['Strategy_Cumulative'], df['Market_Cumulative'],
        where=(df['Strategy_Cumulative'] <  df['Market_Cumulative']),
        alpha=0.12, color=RED, interpolate=True)
    ax2.set_ylabel('Portfolio Value (USD)', fontsize=9)
    ax2.legend(loc='upper left', framealpha=0.3, fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelbottom=False)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    cum  = df['Strategy_Cumulative'].values
    peak = np.maximum.accumulate(cum)
    dd   = np.where(peak > 0, (cum - peak) / peak * 100, 0)
    ax3.fill_between(df.index, dd, 0, color=RED, alpha=0.45)
    ax3.plot(df.index, dd, color=RED, lw=0.8)
    ax3.axhline(y=metrics['max_drawdown'], color=YELLOW, ls='--', lw=1, alpha=0.7,
                label=f"Max DD: {metrics['max_drawdown']:.1f}%")
    ax3.set_ylabel('Drawdown (%)', fontsize=9)
    ax3.set_xlabel('Date', fontsize=9)
    ax3.legend(loc='lower left', framealpha=0.3, fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))

    fig.text(0.99, 0.005, 'QuantEdge v2.4  ·  Research only. Not financial advice.',
             ha='right', va='bottom', fontsize=7, color='#484f58', style='italic')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=220, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY
# ══════════════════════════════════════════════════════════════════════════════

def run_full_backtest(ticker='AAPL', start='2020-01-01', end=None,
                      strategy='ma_crossover', params=None, capital=100_000.0,
                      ohlcv=None):
    if params is None:
        params = {}

    if ohlcv and len(ohlcv) >= 60:
        df = pd.DataFrame(ohlcv)
        df.index = pd.to_datetime(df['date'])
        df.index.name = 'Date'
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        })
        keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
        df = df[keep].copy()
        for col in keep:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Close'])
        df = df.sort_index()
    else:
        df = fetch_data(ticker, start, end)

    df      = add_indicators(df, params)
    df      = STRATEGY_MAP.get(strategy, strat_ma)(df, params)
    df      = run_backtest(df, capital)
    metrics = compute_metrics(df, capital)
    label   = params.get('strategy_name', STRATEGY_LABELS.get(strategy, strategy))
    chart   = generate_chart(df, ticker, metrics, label)

    bis = df[df['Signal'] == 1].index.tolist()
    sis = df[df['Signal'] == -1].index.tolist()
    trades = []
    for i, bd in enumerate(bis):
        fut = [s for s in sis if s > bd]
        if fut:
            sd  = fut[0]
            bp  = round(float(df.loc[bd, 'Close']), 2)
            sp  = round(float(df.loc[sd, 'Close']), 2)
            pnl = round((sp - bp) / bp * 100, 2)
            trades.append({
                "trade":      i + 1,
                "buy_date":   str(bd.date()),
                "buy_price":  bp,
                "sell_date":  str(sd.date()),
                "sell_price": sp,
                "pnl_pct":    pnl,
                "result":     "WIN" if pnl > 0 else "LOSS"
            })

    return {
        "ticker":         ticker,
        "start":          start,
        "end":            end or datetime.today().strftime('%Y-%m-%d'),
        "strategy":       strategy,
        "strategy_label": label,
        "params":         params,
        "metrics":        metrics,
        "chart":          chart,
        "trades":         trades[-10:],
        "data_points":    len(df),
    }