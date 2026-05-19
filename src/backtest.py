"""
=============================================================================
  QuantEdge Backtest Engine v2.0
  Multi-Strategy | Agentic LLM Pipeline | Quant Research Prototype
  Strategies: MA Crossover, RSI, MACD, Bollinger Bands, Custom (LLM-parsed)
=============================================================================
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
import warnings, io, base64, json
import time
warnings.filterwarnings('ignore')

def fetch_data(ticker, start, end):

    import yfinance as yf
    import pandas as pd
    import time

    ticker = ticker.strip().split(" ")[0]

    time.sleep(1)

    ticker_obj = yf.Ticker(ticker)

    raw = ticker_obj.history(
        period="5y",
        auto_adjust=True,
        timeout=20
    )

    print(raw.head())

    if raw.empty:
        raise ValueError(f"No data for {ticker}.")

    df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    df.dropna(inplace=True)

    df.index = pd.to_datetime(df.index)

    return df

def add_indicators(df, params):
    df = df.copy()
    s = params
    df['MA_Short'] = df['Close'].rolling(window=s.get('short_ma',20), min_periods=1).mean()
    df['MA_Long']  = df['Close'].rolling(window=s.get('long_ma', 50), min_periods=1).mean()
    rsi_p = s.get('rsi_period', 14)
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(rsi_p).mean()
    loss  = (-delta.clip(upper=0)).rolling(rsi_p).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100/(1+rs))
    ema_fast = df['Close'].ewm(span=s.get('macd_fast',12), adjust=False).mean()
    ema_slow = df['Close'].ewm(span=s.get('macd_slow',26), adjust=False).mean()
    df['MACD']        = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=s.get('macd_signal',9), adjust=False).mean()
    bb_p = s.get('bb_period',20); bb_s = s.get('bb_std',2.0)
    df['BB_Mid']   = df['Close'].rolling(bb_p).mean()
    df['BB_Std']   = df['Close'].rolling(bb_p).std()
    df['BB_Upper'] = df['BB_Mid'] + bb_s * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - bb_s * df['BB_Std']
    high_low   = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close  = (df['Low']  - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    return df

def strategy_ma_crossover(df, params):
    df = df.copy()
    lm = params.get('long_ma',50)
    df['Position'] = np.where(df['MA_Short'] > df['MA_Long'], 1, 0)
    df['Signal']   = df['Position'].diff()
    df.loc[df.index[:lm], ['Signal','Position']] = 0
    return df

def strategy_rsi(df, params):
    df = df.copy(); ob=params.get('rsi_overbought',70); os_=params.get('rsi_oversold',30)
    pos=[]; cur=0
    for rsi in df['RSI']:
        if pd.isna(rsi): pos.append(0); continue
        if rsi < os_: cur=1
        elif rsi > ob: cur=0
        pos.append(cur)
    df['Position']=pos; df['Signal']=pd.Series(pos,index=df.index).diff()
    return df

def strategy_macd(df, params):
    df = df.copy()
    df['Position'] = np.where(df['MACD'] > df['MACD_Signal'], 1, 0)
    df['Signal']   = df['Position'].diff()
    df.loc[df.index[:30], ['Signal','Position']] = 0
    return df

def strategy_bollinger(df, params):
    df=df.copy(); pos=[]; cur=0
    for _,row in df.iterrows():
        if pd.isna(row['BB_Upper']): pos.append(0); continue
        if row['Close'] <= row['BB_Lower']: cur=1
        elif row['Close'] >= row['BB_Upper']: cur=0
        pos.append(cur)
    df['Position']=pos; df['Signal']=pd.Series(pos,index=df.index).diff()
    return df

def strategy_combined(df, params):
    df=df.copy(); ob=params.get('rsi_overbought',70); lm=params.get('long_ma',50)
    ma_long = df['MA_Short']>df['MA_Long']; rsi_ok = df['RSI']<ob
    pos=[]; cur=0
    for i in range(len(df)):
        if i<lm: pos.append(0); continue
        if ma_long.iloc[i] and rsi_ok.iloc[i]: cur=1
        elif not ma_long.iloc[i]: cur=0
        pos.append(cur)
    df['Position']=pos; df['Signal']=pd.Series(pos,index=df.index).diff()
    return df

def strategy_custom_parsed(df, params):
    rules = params.get('custom_rules', {})
    buy_rule  = rules.get('buy',  'ma_crossover_up')
    sell_rule = rules.get('sell', 'ma_crossover_down')
    df=df.copy(); pos=[]; cur=0
    for _,row in df.iterrows():
        if pd.isna(row['MA_Short']): pos.append(0); continue
        buy_ok = False; sell_ok = False
        if buy_rule=='ma_crossover_up':       buy_ok = row['MA_Short']>row['MA_Long']
        elif buy_rule=='rsi_oversold':         buy_ok = not pd.isna(row['RSI']) and row['RSI']<params.get('rsi_oversold',30)
        elif buy_rule=='macd_crossover_up':    buy_ok = not pd.isna(row['MACD']) and row['MACD']>row['MACD_Signal']
        elif buy_rule=='bb_lower_touch':       buy_ok = not pd.isna(row['BB_Lower']) and row['Close']<=row['BB_Lower']
        elif buy_rule=='ma_and_rsi':           buy_ok = row['MA_Short']>row['MA_Long'] and not pd.isna(row['RSI']) and row['RSI']<60
        if sell_rule=='ma_crossover_down':     sell_ok = row['MA_Short']<row['MA_Long']
        elif sell_rule=='rsi_overbought':      sell_ok = not pd.isna(row['RSI']) and row['RSI']>params.get('rsi_overbought',70)
        elif sell_rule=='macd_crossover_down': sell_ok = not pd.isna(row['MACD']) and row['MACD']<row['MACD_Signal']
        elif sell_rule=='bb_upper_touch':      sell_ok = not pd.isna(row['BB_Upper']) and row['Close']>=row['BB_Upper']
        elif sell_rule=='ma_or_rsi':           sell_ok = row['MA_Short']<row['MA_Long'] or (not pd.isna(row['RSI']) and row['RSI']>70)
        if buy_ok: cur=1
        if sell_ok: cur=0
        pos.append(cur)
    df['Position']=pos; df['Signal']=pd.Series(pos,index=df.index).diff()
    return df

STRATEGY_MAP = {
    'ma_crossover': strategy_ma_crossover,
    'rsi':          strategy_rsi,
    'macd':         strategy_macd,
    'bollinger':    strategy_bollinger,
    'combined':     strategy_combined,
    'custom':       strategy_custom_parsed,
}

def run_backtest(df, initial_capital=100_000.0):
    df=df.copy()
    df['Market_Return']   = np.log(df['Close']/df['Close'].shift(1))
    df['Strategy_Return'] = df['Market_Return']*df['Position'].shift(1)
    df['Market_Cumulative']   = initial_capital*np.exp(df['Market_Return'].cumsum())
    df['Strategy_Cumulative'] = initial_capital*np.exp(df['Strategy_Return'].cumsum())
    df.ffill(inplace=True); df.fillna(0,inplace=True)
    return df

def compute_metrics(df, initial_capital=100_000.0, risk_free_rate=0.05):
    sr = df['Strategy_Return'].dropna()
    fv = df['Strategy_Cumulative'].iloc[-1]
    mf = df['Market_Cumulative'].iloc[-1]
    tr = (fv - initial_capital) / initial_capital * 100
    mr = (mf - initial_capital) / initial_capital * 100
    ny = len(df) / 252
    cagr = ((fv / initial_capital) ** (1 / ny) - 1) * 100 if ny > 0 else 0

    # ✅ FIX: Sharpe ratio — guard against zero/near-zero std (avoids -111256... overflow)
    ex = sr - (risk_free_rate / 252)
    std = ex.std()
    if std is None or np.isnan(std) or std < 1e-10:
        sharpe = 0.0
    else:
        sharpe = round(float((ex.mean() / std) * np.sqrt(252)), 3)
        # Clamp to sane range — anything beyond ±50 is a data artifact
        sharpe = max(-50.0, min(50.0, sharpe))

    vol = sr.std() * np.sqrt(252) * 100
    cum = df['Strategy_Cumulative']
    rm  = cum.cummax()
    dd  = (cum - rm) / rm * 100
    maxdd = dd.min()

    bis = df[df['Signal'] == 1].index
    sis = df[df['Signal'] == -1].index
    trades = []; wins = []
    for bd in bis:
        fut = sis[sis > bd]
        if len(fut) > 0:
            sd  = fut[0]
            bp  = float(df.loc[bd, 'Close'])
            sp  = float(df.loc[sd, 'Close'])
            pnl = sp - bp
            trades.append(pnl)
            wins.append(1 if pnl > 0 else 0)

    wr     = (sum(wins) / len(wins) * 100) if wins else 0
    calmar = (cagr / abs(maxdd)) if maxdd != 0 else 0

    return {
        "total_return":  round(tr, 2),
        "market_return": round(mr, 2),
        "cagr":          round(cagr, 2),
        "sharpe_ratio":  sharpe,
        "max_drawdown":  round(maxdd, 2),
        "volatility":    round(vol, 2),
        "win_rate":      round(wr, 2),
        "n_trades":      len(trades),
        "avg_trade_pnl": round(np.mean(trades) if trades else 0, 2),
        "calmar_ratio":  round(calmar, 3),
        "final_value":   round(fv, 2),
        "initial_capital": initial_capital,
    }

def generate_charts(df, ticker, metrics, strategy_name='MA Crossover'):
    BG='#0d1117';PANEL='#161b22';GREEN='#39d353';RED='#f85149';BLUE='#58a6ff'
    YELLOW='#e3b341';GRAY='#8b949e';WHITE='#f0f6fc';ACCENT='#1f6feb'
    plt.rcParams.update({'font.family':'monospace','text.color':WHITE,'axes.labelcolor':GRAY,
        'xtick.color':GRAY,'ytick.color':GRAY,'figure.facecolor':BG,'axes.facecolor':PANEL,
        'axes.edgecolor':'#30363d','grid.color':'#21262d','grid.linewidth':0.5})
    fig=plt.figure(figsize=(16,14),facecolor=BG)
    gs=GridSpec(3,1,figure=fig,hspace=0.08,height_ratios=[3,2,1.5])
    ax1=fig.add_subplot(gs[0]); ax2=fig.add_subplot(gs[1]); ax3=fig.add_subplot(gs[2])
    ax1.plot(df.index,df['Close'],color=WHITE,lw=1.2,alpha=0.9,label='Close')
    ax1.plot(df.index,df['MA_Short'],color=BLUE,lw=1.5,ls='--',label='MA-Short')
    ax1.plot(df.index,df['MA_Long'],color=YELLOW,lw=1.5,ls='--',label='MA-Long')
    buys=df[df['Signal']==1]; sells=df[df['Signal']==-1]
    ax1.scatter(buys.index,buys['Close'],marker='^',color=GREEN,s=120,zorder=5,label=f'Buy ({len(buys)})')
    ax1.scatter(sells.index,sells['Close'],marker='v',color=RED,s=120,zorder=5,label=f'Sell ({len(sells)})')
    ax1.fill_between(df.index,df['Close'].min(),df['Close'].max(),where=df['Position']==1,alpha=0.05,color=GREEN)
    ax1.set_title(f'  {ticker} | {strategy_name}  ·  QuantEdge v2.0',color=WHITE,fontsize=13,fontweight='bold',loc='left',pad=12)
    ax1.legend(loc='upper left',framealpha=0.3,fontsize=8,ncol=6)
    ax1.set_ylabel('Price (USD)',fontsize=9); ax1.grid(True,alpha=0.3); ax1.tick_params(labelbottom=False)
    ann=(f"Return: {metrics['total_return']:+.1f}%  |  Sharpe: {metrics['sharpe_ratio']:.2f}  |  "
         f"Max DD: {metrics['max_drawdown']:.1f}%  |  Win: {metrics['win_rate']:.1f}%  |  Trades: {metrics['n_trades']}")
    ax1.annotate(ann,xy=(0.01,0.02),xycoords='axes fraction',fontsize=8.5,color=GRAY,
                 bbox=dict(boxstyle='round,pad=0.4',facecolor='#0d1117',alpha=0.7))
    ax2.plot(df.index,df['Strategy_Cumulative'],color=GREEN,lw=2,label='Strategy')
    ax2.plot(df.index,df['Market_Cumulative'],color=ACCENT,lw=1.5,label='Buy & Hold',alpha=0.7)
    ax2.axhline(y=metrics['initial_capital'],color=GRAY,ls=':',lw=0.8,alpha=0.5)
    ax2.fill_between(df.index,df['Strategy_Cumulative'],df['Market_Cumulative'],
        where=df['Strategy_Cumulative']>=df['Market_Cumulative'],alpha=0.12,color=GREEN,interpolate=True)
    ax2.fill_between(df.index,df['Strategy_Cumulative'],df['Market_Cumulative'],
        where=df['Strategy_Cumulative']<df['Market_Cumulative'],alpha=0.12,color=RED,interpolate=True)
    ax2.set_ylabel('Portfolio Value (USD)',fontsize=9); ax2.legend(loc='upper left',framealpha=0.3,fontsize=9)
    ax2.grid(True,alpha=0.3); ax2.tick_params(labelbottom=False)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'${x:,.0f}'))
    dd=(df['Strategy_Cumulative']-df['Strategy_Cumulative'].cummax())/df['Strategy_Cumulative'].cummax()*100
    ax3.fill_between(df.index,dd,0,color=RED,alpha=0.45)
    ax3.plot(df.index,dd,color=RED,lw=0.8)
    ax3.axhline(y=metrics['max_drawdown'],color=YELLOW,ls='--',lw=1,alpha=0.7,label=f"Max DD: {metrics['max_drawdown']:.1f}%")
    ax3.set_ylabel('Drawdown (%)',fontsize=9); ax3.set_xlabel('Date',fontsize=9)
    ax3.legend(loc='lower left',framealpha=0.3,fontsize=9); ax3.grid(True,alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f'{x:.0f}%'))
    fig.text(0.99,0.005,'QuantEdge v2.0  ·  Research only. Not financial advice.',
             ha='right',va='bottom',fontsize=7,color='#484f58',style='italic')
    buf=io.BytesIO()
    plt.savefig(buf,format='png',dpi=150,bbox_inches='tight',facecolor=BG)
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

STRATEGY_LABELS = {
    'ma_crossover':'MA Crossover','rsi':'RSI Mean-Reversion',
    'macd':'MACD Crossover','bollinger':'Bollinger Bands',
    'combined':'MA + RSI Combined','custom':'Custom Strategy',
}

def run_full_backtest(ticker='AAPL', start='2020-01-01', end=None,
                      strategy='ma_crossover', params=None, capital=100_000.0):
    if end is None: end=datetime.today().strftime('%Y-%m-%d')
    if params is None: params={}
    df=fetch_data(ticker,start,end)


    
    df=add_indicators(df,params)
    df=STRATEGY_MAP.get(strategy,strategy_ma_crossover)(df,params)
    df=run_backtest(df,initial_capital=capital)
    metrics=compute_metrics(df,initial_capital=capital)
    label=params.get('strategy_name',STRATEGY_LABELS.get(strategy,strategy))
    chart=generate_charts(df,ticker,metrics,label)
    bis=df[df['Signal']==1].index.tolist(); sis=df[df['Signal']==-1].index.tolist()
    trades=[]
    for i,bd in enumerate(bis):
        fut=[s for s in sis if s>bd]
        if fut:
            sd=fut[0]; bp=round(float(df.loc[bd,'Close']),2); sp=round(float(df.loc[sd,'Close']),2)
            pnl=round((sp-bp)/bp*100,2)
            trades.append({"trade":i+1,"buy_date":str(bd.date()),"buy_price":bp,
                           "sell_date":str(sd.date()),"sell_price":sp,"pnl_pct":pnl,
                           "result":"WIN" if pnl>0 else "LOSS"})
    return {"ticker":ticker,"start":start,"end":end,"strategy":strategy,
            "strategy_label":label,"params":params,"metrics":metrics,
            "chart":chart,"trades":trades[-10:],"data_points":len(df)}

if __name__=='__main__':
    print("\n"+"="*65+"\n  QuantEdge v2.0 — Multi-Strategy Test\n"+"="*65)
    for strat in ['ma_crossover','rsi','macd','bollinger','combined']:
        r=run_full_backtest('AAPL','2020-01-01',strategy=strat)
        m=r['metrics']
        print(f"\n  [{r['strategy_label']}]")
        print(f"  Return:{m['total_return']:+.1f}%  Sharpe:{m['sharpe_ratio']:.3f}  MaxDD:{m['max_drawdown']:.1f}%  Trades:{m['n_trades']}")
    print("\n"+"="*65)