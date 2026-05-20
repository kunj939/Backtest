"""
QuantEdge Backtest Engine v2.2
Data sources: Stooq (primary, no API key) → yfinance fallback
"""
import pandas as pd
import numpy as np
import urllib.request
import io as _io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
import warnings, io, base64
warnings.filterwarnings('ignore')


# ── DATA FETCH ────────────────────────────────────────────────────────────────
def _clean_df(raw, ticker):
    """Normalise columns, strip timezone, validate."""
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            raw = raw.xs(ticker.upper(), axis=1, level=1)
        except Exception:
            raw.columns = raw.columns.get_level_values(0)

    raw.columns = [c.strip().title() for c in raw.columns]
    cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in raw.columns]
    if 'Close' not in cols:
        return None
    df = raw[cols].copy()
    df.dropna(subset=['Close'], inplace=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.sort_index(inplace=True)
    return df if len(df) >= 60 else None


def _fetch_stooq(ticker, start, end):
    """
    Stooq CSV API — free, no key, works on Vercel.
    URL: https://stooq.com/q/d/l/?s=AAPL.US&d1=20200101&d2=20251231&i=d
    """
    # Stooq uses lowercase ticker + ".US" suffix for US stocks
    sym = ticker.lower()
    if '.' not in sym:
        sym = sym + '.us'
    d1 = start.replace('-', '')
    d2 = end.replace('-', '')
    url = f"https://stooq.com/q/d/l/?s={sym}&d1={d1}&d2={d2}&i=d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        content = r.read().decode('utf-8')
    if 'No data' in content or len(content) < 100:
        return None
    df = pd.read_csv(_io.StringIO(content), parse_dates=['Date'], index_col='Date')
    df.columns = [c.strip().title() for c in df.columns]
    df.sort_index(inplace=True)
    df.dropna(subset=['Close'], inplace=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df if len(df) >= 60 else None


def _fetch_alphavantage(ticker, start, end):
    """
    Alpha Vantage free tier — requires ALPHA_VANTAGE_KEY env var.
    Get free key at: https://www.alphavantage.co/support/#api-key
    """
    import os, json
    key = os.environ.get('ALPHA_VANTAGE_KEY', '')
    if not key:
        return None
    url = (f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
           f"&symbol={ticker}&outputsize=full&apikey={key}&datatype=csv")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        content = r.read().decode('utf-8')
    if 'timestamp' not in content:
        return None
    df = pd.read_csv(_io.StringIO(content), parse_dates=['timestamp'], index_col='timestamp')
    df.index.name = 'Date'
    df = df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'adjusted_close': 'Close', 'volume': 'Volume'
    })
    df.sort_index(inplace=True)
    df = df[df.index >= pd.to_datetime(start)]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.dropna(subset=['Close'], inplace=True)
    return df if len(df) >= 60 else None


    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        raw = t.history(start=start, end=end, auto_adjust=True)
        if raw is None or raw.empty:
            raw = t.history(period="5y", auto_adjust=True)
        return _clean_df(raw, ticker) if (raw is not None and not raw.empty) else None
    except Exception:
        return None


def fetch_data(ticker, start, end=None):
    ticker = ticker.strip().upper().split()[0]
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    df = None

    # ── Source 1: Stooq (free CSV, no API key needed) ────────────────────────
    try:
        df = _fetch_stooq(ticker, start, end)
    except Exception:
        pass

    # ── Source 2: Alpha Vantage (free, needs ALPHA_VANTAGE_KEY in .env) ──────
    if df is None:
        try:
            df = _fetch_alphavantage(ticker, start, end)
        except Exception:
            pass

    # ── Source 3: yfinance fallback ──────────────────────────────────────────
    if df is None:
        df = _fetch_yfinance(ticker, start, end)

    if df is None:
        raise ValueError(
            f"No data found for '{ticker}'. "
            "Check the ticker symbol (use US symbols like AAPL, TSLA, MSFT)."
        )

    cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
    return df[cols]


# ── INDICATORS ────────────────────────────────────────────────────────────────
def add_indicators(df, p):
    df = df.copy()
    df['MA_Short'] = df['Close'].rolling(p.get('short_ma', 20), min_periods=1).mean()
    df['MA_Long']  = df['Close'].rolling(p.get('long_ma',  50), min_periods=1).mean()

    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(p.get('rsi_period', 14)).mean()
    loss  = (-delta.clip(upper=0)).rolling(p.get('rsi_period', 14)).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    ema_f = df['Close'].ewm(span=p.get('macd_fast', 12), adjust=False).mean()
    ema_s = df['Close'].ewm(span=p.get('macd_slow', 26), adjust=False).mean()
    df['MACD']        = ema_f - ema_s
    df['MACD_Signal'] = df['MACD'].ewm(span=p.get('macd_signal', 9), adjust=False).mean()

    bp, bs = p.get('bb_period', 20), p.get('bb_std', 2.0)
    df['BB_Mid']   = df['Close'].rolling(bp).mean()
    df['BB_Std']   = df['Close'].rolling(bp).std()
    df['BB_Upper'] = df['BB_Mid'] + bs * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - bs * df['BB_Std']

    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low']  - df['Close'].shift()).abs()
    df['ATR'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    return df


# ── STRATEGIES ────────────────────────────────────────────────────────────────
def strat_ma(df, p):
    df = df.copy()
    lm = p.get('long_ma', 50)
    df['Position'] = np.where(df['MA_Short'] > df['MA_Long'], 1, 0)
    df['Signal']   = df['Position'].diff()
    df.loc[df.index[:lm], ['Signal', 'Position']] = 0
    return df

def strat_rsi(df, p):
    df = df.copy()
    pos, cur = [], 0
    for v in df['RSI']:
        if pd.isna(v): pos.append(0); continue
        if v < p.get('rsi_oversold', 30):   cur = 1
        elif v > p.get('rsi_overbought', 70): cur = 0
        pos.append(cur)
    df['Position'] = pos
    df['Signal']   = pd.Series(pos, index=df.index).diff()
    return df

def strat_macd(df, p):
    df = df.copy()
    df['Position'] = np.where(df['MACD'] > df['MACD_Signal'], 1, 0)
    df['Signal']   = df['Position'].diff()
    df.loc[df.index[:30], ['Signal', 'Position']] = 0
    return df

def strat_bb(df, p):
    df = df.copy()
    pos, cur = [], 0
    for _, row in df.iterrows():
        if pd.isna(row['BB_Upper']): pos.append(0); continue
        if row['Close'] <= row['BB_Lower']:  cur = 1
        elif row['Close'] >= row['BB_Upper']: cur = 0
        pos.append(cur)
    df['Position'] = pos
    df['Signal']   = pd.Series(pos, index=df.index).diff()
    return df

def strat_combined(df, p):
    df = df.copy()
    lm = p.get('long_ma', 50)
    pos, cur = [], 0
    for i in range(len(df)):
        if i < lm: pos.append(0); continue
        ma_ok  = df['MA_Short'].iloc[i] > df['MA_Long'].iloc[i]
        rsi_ok = df['RSI'].iloc[i] < p.get('rsi_overbought', 70)
        if ma_ok and rsi_ok:  cur = 1
        elif not ma_ok:       cur = 0
        pos.append(cur)
    df['Position'] = pos
    df['Signal']   = pd.Series(pos, index=df.index).diff()
    return df

def strat_custom(df, p):
    rules     = p.get('custom_rules', {})
    buy_rule  = rules.get('buy',  'ma_crossover_up')
    sell_rule = rules.get('sell', 'ma_crossover_down')
    df = df.copy()
    pos, cur = [], 0
    for _, row in df.iterrows():
        if pd.isna(row['MA_Short']): pos.append(0); continue
        buy = sell = False
        if buy_rule == 'ma_crossover_up':     buy = row['MA_Short'] > row['MA_Long']
        elif buy_rule == 'rsi_oversold':      buy = not pd.isna(row['RSI']) and row['RSI'] < p.get('rsi_oversold', 30)
        elif buy_rule == 'macd_crossover_up': buy = not pd.isna(row['MACD']) and row['MACD'] > row['MACD_Signal']
        elif buy_rule == 'bb_lower_touch':    buy = not pd.isna(row['BB_Lower']) and row['Close'] <= row['BB_Lower']
        elif buy_rule == 'ma_and_rsi':        buy = row['MA_Short'] > row['MA_Long'] and not pd.isna(row['RSI']) and row['RSI'] < 60
        if sell_rule == 'ma_crossover_down':     sell = row['MA_Short'] < row['MA_Long']
        elif sell_rule == 'rsi_overbought':      sell = not pd.isna(row['RSI']) and row['RSI'] > p.get('rsi_overbought', 70)
        elif sell_rule == 'macd_crossover_down': sell = not pd.isna(row['MACD']) and row['MACD'] < row['MACD_Signal']
        elif sell_rule == 'bb_upper_touch':      sell = not pd.isna(row['BB_Upper']) and row['Close'] >= row['BB_Upper']
        elif sell_rule == 'ma_or_rsi':           sell = row['MA_Short'] < row['MA_Long'] or (not pd.isna(row['RSI']) and row['RSI'] > 70)
        if buy:  cur = 1
        if sell: cur = 0
        pos.append(cur)
    df['Position'] = pos
    df['Signal']   = pd.Series(pos, index=df.index).diff()
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


# ── BACKTEST + METRICS ────────────────────────────────────────────────────────
def run_backtest(df, capital):
    df = df.copy()
    df['Market_Return']   = np.log(df['Close'] / df['Close'].shift(1))
    df['Strategy_Return'] = df['Market_Return'] * df['Position'].shift(1)
    df['Market_Cumulative']   = capital * np.exp(df['Market_Return'].cumsum())
    df['Strategy_Cumulative'] = capital * np.exp(df['Strategy_Return'].cumsum())
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    return df

def compute_metrics(df, capital, rfr=0.05):
    sr = df['Strategy_Return'].dropna()
    fv = float(df['Strategy_Cumulative'].iloc[-1])
    mf = float(df['Market_Cumulative'].iloc[-1])
    ny = max(len(df) / 252, 0.01)

    tr   = (fv - capital) / capital * 100
    mr   = (mf - capital) / capital * 100
    cagr = ((fv / capital) ** (1 / ny) - 1) * 100

    ex  = sr - (rfr / 252)
    std = float(ex.std())
    sharpe = round(max(-50.0, min(50.0, float((ex.mean() / std) * np.sqrt(252)))), 3) if std > 1e-10 else 0.0

    vol   = float(sr.std() * np.sqrt(252) * 100)
    cum   = df['Strategy_Cumulative']
    dd    = (cum - cum.cummax()) / cum.cummax() * 100
    maxdd = float(dd.min())

    bis, sis = df[df['Signal'] == 1].index, df[df['Signal'] == -1].index
    trades, wins = [], []
    for bd in bis:
        fut = sis[sis > bd]
        if len(fut):
            sd = fut[0]
            pnl = float(df.loc[sd, 'Close']) - float(df.loc[bd, 'Close'])
            trades.append(pnl); wins.append(1 if pnl > 0 else 0)

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


# ── CHART ─────────────────────────────────────────────────────────────────────
def generate_chart(df, ticker, metrics, label):
    BG='#0d1117'; PANEL='#161b22'; GREEN='#39d353'; RED='#f85149'
    BLUE='#58a6ff'; YELLOW='#e3b341'; GRAY='#8b949e'; WHITE='#f0f6fc'; ACCENT='#1f6feb'
    plt.rcParams.update({'font.family':'monospace','text.color':WHITE,'axes.labelcolor':GRAY,
        'xtick.color':GRAY,'ytick.color':GRAY,'figure.facecolor':BG,'axes.facecolor':PANEL,
        'axes.edgecolor':'#30363d','grid.color':'#21262d','grid.linewidth':0.5})

    fig = plt.figure(figsize=(16, 14), facecolor=BG)
    gs  = GridSpec(3, 1, figure=fig, hspace=0.08, height_ratios=[3, 2, 1.5])
    ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    ax1.plot(df.index, df['Close'],    color=WHITE,  lw=1.2, alpha=0.9, label='Close')
    ax1.plot(df.index, df['MA_Short'], color=BLUE,   lw=1.5, ls='--',   label='MA-Short')
    ax1.plot(df.index, df['MA_Long'],  color=YELLOW, lw=1.5, ls='--',   label='MA-Long')
    buys = df[df['Signal']==1]; sells = df[df['Signal']==-1]
    ax1.scatter(buys.index,  buys['Close'],  marker='^', color=GREEN, s=100, zorder=5, label=f'Buy ({len(buys)})')
    ax1.scatter(sells.index, sells['Close'], marker='v', color=RED,   s=100, zorder=5, label=f'Sell ({len(sells)})')
    ax1.fill_between(df.index, df['Close'].min(), df['Close'].max(), where=df['Position']==1, alpha=0.05, color=GREEN)
    ax1.set_title(f'  {ticker} | {label}  ·  QuantEdge v2.1', color=WHITE, fontsize=13, fontweight='bold', loc='left', pad=12)
    ax1.legend(loc='upper left', framealpha=0.3, fontsize=8, ncol=6)
    ax1.set_ylabel('Price (USD)', fontsize=9); ax1.grid(True, alpha=0.3); ax1.tick_params(labelbottom=False)
    ann = f"Return: {metrics['total_return']:+.1f}%  |  Sharpe: {metrics['sharpe_ratio']:.2f}  |  Max DD: {metrics['max_drawdown']:.1f}%  |  Win: {metrics['win_rate']:.1f}%  |  Trades: {metrics['n_trades']}"
    ax1.annotate(ann, xy=(0.01,0.02), xycoords='axes fraction', fontsize=8.5, color=GRAY,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d1117', alpha=0.7))

    ax2.plot(df.index, df['Strategy_Cumulative'], color=GREEN,  lw=2,   label='Strategy')
    ax2.plot(df.index, df['Market_Cumulative'],   color=ACCENT, lw=1.5, label='Buy & Hold', alpha=0.7)
    ax2.axhline(y=metrics['initial_capital'], color=GRAY, ls=':', lw=0.8, alpha=0.5)
    ax2.fill_between(df.index, df['Strategy_Cumulative'], df['Market_Cumulative'],
        where=df['Strategy_Cumulative']>=df['Market_Cumulative'], alpha=0.12, color=GREEN, interpolate=True)
    ax2.fill_between(df.index, df['Strategy_Cumulative'], df['Market_Cumulative'],
        where=df['Strategy_Cumulative']<df['Market_Cumulative'],  alpha=0.12, color=RED,   interpolate=True)
    ax2.set_ylabel('Portfolio Value (USD)', fontsize=9); ax2.legend(loc='upper left', framealpha=0.3, fontsize=9)
    ax2.grid(True, alpha=0.3); ax2.tick_params(labelbottom=False)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x:,.0f}'))

    dd = (df['Strategy_Cumulative'] - df['Strategy_Cumulative'].cummax()) / df['Strategy_Cumulative'].cummax() * 100
    ax3.fill_between(df.index, dd, 0, color=RED, alpha=0.45)
    ax3.plot(df.index, dd, color=RED, lw=0.8)
    ax3.axhline(y=metrics['max_drawdown'], color=YELLOW, ls='--', lw=1, alpha=0.7, label=f"Max DD: {metrics['max_drawdown']:.1f}%")
    ax3.set_ylabel('Drawdown (%)', fontsize=9); ax3.set_xlabel('Date', fontsize=9)
    ax3.legend(loc='lower left', framealpha=0.3, fontsize=9); ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:.0f}%'))

    fig.text(0.99, 0.005, 'QuantEdge v2.1  ·  Research only. Not financial advice.',
             ha='right', va='bottom', fontsize=7, color='#484f58', style='italic')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ── MAIN ENTRY ────────────────────────────────────────────────────────────────
def run_full_backtest(ticker='AAPL', start='2020-01-01', end=None,
                      strategy='ma_crossover', params=None, capital=100_000.0):
    if params is None: params = {}
    df      = fetch_data(ticker, start, end)
    df      = add_indicators(df, params)
    df      = STRATEGY_MAP.get(strategy, strat_ma)(df, params)
    df      = run_backtest(df, capital)
    metrics = compute_metrics(df, capital)
    label   = params.get('strategy_name', STRATEGY_LABELS.get(strategy, strategy))
    chart   = generate_chart(df, ticker, metrics, label)

    bis = df[df['Signal']==1].index.tolist()
    sis = df[df['Signal']==-1].index.tolist()
    trades = []
    for i, bd in enumerate(bis):
        fut = [s for s in sis if s > bd]
        if fut:
            sd = fut[0]
            bp = round(float(df.loc[bd,'Close']), 2)
            sp = round(float(df.loc[sd,'Close']), 2)
            pnl = round((sp-bp)/bp*100, 2)
            trades.append({"trade":i+1,"buy_date":str(bd.date()),"buy_price":bp,
                           "sell_date":str(sd.date()),"sell_price":sp,
                           "pnl_pct":pnl,"result":"WIN" if pnl>0 else "LOSS"})
    return {
        "ticker": ticker, "start": start,
        "end": end or datetime.today().strftime('%Y-%m-%d'),
        "strategy": strategy, "strategy_label": label,
        "params": params, "metrics": metrics,
        "chart": chart, "trades": trades[-10:], "data_points": len(df),
    }