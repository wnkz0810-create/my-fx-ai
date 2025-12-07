import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
import optuna

# ページの設定
st.set_page_config(page_title="Multi-Asset AI Quant System", layout="wide")

st.title("🤖 Multi-Asset AI Quant System (Final Ver.)")
st.write("予測、バックテスト、最適化に加え、**「適切な取引量の自動管理」**機能を搭載した完全版です。")

# --- サイドバー設定 (銘柄選択) ---
st.sidebar.header("銘柄選択")

pair_options = {
    "Gold (金先物 GC=F)": "GC=F", 
    "USD/JPY (ドル円)": "JPY=X",
    "EUR/USD (ユーロドル)": "EURUSD=X",
    "BTC/USD (ビットコイン)": "BTC-USD"
}

selected_label = st.sidebar.selectbox("トレードする銘柄", list(pair_options.keys()))
ticker = pair_options[selected_label]

# --- 通貨ごとのデフォルト設定 (閾値・取引量など) ---
if ticker == "GC=F" or ticker == "XAUUSD=X":
    # ゴールド用
    def_th = 2.00   
    def_tp = 10.00  
    def_sl = 5.00   
    num_step = 0.10 
    curr_unit = "$"
    # ★重要: ゴールドはボラティリティが高いので取引量を落とす
    default_amount = 100.0 
    spread_cost = 0.30 
elif ticker == "BTC-USD":
    # ビットコイン用
    def_th = 100.0
    def_tp = 500.0
    def_sl = 300.0
    num_step = 10.0
    curr_unit = "$"
    default_amount = 0.1 
    spread_cost = 50.0
elif ticker == "JPY=X":
    # ドル円用
    def_th = 0.050
    def_tp = 0.500
    def_sl = 0.200
    num_step = 0.001
    curr_unit = "円"
    default_amount = 10000.0
    spread_cost = 0.003
else:
    # その他 (ユーロドルなど)
    def_th = 0.0010
    def_tp = 0.0050
    def_sl = 0.0020
    num_step = 0.0001
    curr_unit = "pips"
    default_amount = 10000.0
    spread_cost = 0.0003

# --- 取引設定 (サイドバー) ---
st.sidebar.markdown("---")
st.sidebar.header("取引設定")

# 取引量の入力 (警告対策: formatを%.0fに変更)
trade_amount = st.sidebar.number_input(
    "1回の取引量 (Lot/Unit)", 
    min_value=0.01, 
    max_value=1000000.0, 
    value=float(default_amount), 
    step=100.0 if default_amount >= 100 else 0.1,
    format="%.2f" if default_amount < 10 else "%.0f", # ここを修正しました
    help="ゴールドなら100、ドル円なら10000などが目安です。"
)

st.sidebar.markdown("---")
st.sidebar.header(f"AIパラメータ ({curr_unit})")

p_threshold = st.sidebar.number_input(
    f"エントリー閾値", 0.0000, 1000.0000, def_th, step=num_step, format="%.4f", key=f"th_{ticker}"
)

st.sidebar.subheader("リスク管理")
p_tp = st.sidebar.number_input(
    f"利確幅 TP", 0.0000, 2000.0000, def_tp, step=num_step, format="%.4f", key=f"tp_{ticker}"
)
p_sl = st.sidebar.number_input(
    f"損切幅 SL", 0.0000, 1000.0000, def_sl, step=num_step, format="%.4f", key=f"sl_{ticker}"
)

p_n_est = st.sidebar.number_input("決定木の数", 10, 300, 100)
p_sma_s = st.sidebar.number_input("短期SMA期間", 2, 20, 5)
p_sma_l = st.sidebar.number_input("長期SMA期間", 20, 100, 25)

params = {
    "threshold": p_threshold,
    "tp": p_tp,
    "sl": p_sl,
    "sma_short": p_sma_s,
    "sma_long": p_sma_l,
    "n_estimators": p_n_est
}

# --- データ取得関数 (エラーハンドリング強化版) ---
@st.cache_data(ttl=3600)
def get_historical_data(ticker_symbol, period="2y", interval="1h"):
    try:
        df = yf.download(ticker_symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return pd.DataFrame() # 空のDataFrameを返す

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df[['Open', 'High', 'Low', 'Close']].rename(columns={
            'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close'
        })
        return df
    except Exception:
        return pd.DataFrame()

def get_realtime_price(ticker_symbol):
    try:
        data = yf.download(ticker_symbol, period="1d", interval="1m", progress=False)
        if len(data) > 0:
            return float(data['Close'].iloc[-1])
    except:
        return None
    return None

# --- バックテストロジック (取引量対応版) ---
def run_backtest_logic(df_original, params, test_period_days, spread_cost, trade_amount):
    if df_original is None or df_original.empty:
        return None

    df = df_original.copy()
    
    sma_s = params.get('sma_short', 5)
    sma_l = params.get('sma_long', 25)
    threshold = params.get('threshold', 0.05)
    n_estimators = params.get('n_estimators', 100)
    tp_val = params.get('tp', 0.50)
    sl_val = params.get('sl', 0.20)
    
    # 指標計算
    df['SMA_Short'] = df['Close'].rolling(window=sma_s).mean()
    df['SMA_Long'] = df['Close'].rolling(window=sma_l).mean()
    df['Change'] = df['Close'].pct_change()
    df['Std'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_Long'] + (df['Std'] * 2)
    df['Lower_Band'] = df['SMA_Long'] - (df['Std'] * 2)

    df['Next_Close'] = df['Close'].shift(-1)
    df['Next_High'] = df['High'].shift(-1)
    df['Next_Low'] = df['Low'].shift(-1)
    
    df = df.dropna()

    # 安全装置
    if len(df) < 50:
        return None

    features = ['Close', 'SMA_Short', 'SMA_Long', 'Change', 'Upper_Band', 'Lower_Band']
    X = df[features]
    y = df['Next_Close']

    test_rows = test_period_days * 24
    if len(df) < test_rows + 50:
        return None
        
    X_train = X.iloc[:-test_rows]
    y_train = y.iloc[:-test_rows]
    X_test = X.iloc[-test_rows:]
    
    y_test_close = df['Next_Close'].iloc[-test_rows:]
    y_test_high = df['Next_High'].iloc[-test_rows:]
    y_test_low = df['Next_Low'].iloc[-test_rows:]
    price_test = df['Close'].iloc[-test_rows:]

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    balance = 1000000 
    cumulative_profit = []
    dates = []
    total_trades = 0
    wins = 0
    
    for i in range(len(X_test)):
        current_price = price_test.iloc[i]
        pred_price = predictions[i]
        actual_next_close = y_test_close.iloc[i]
        actual_next_high = y_test_high.iloc[i]
        actual_next_low = y_test_low.iloc[i]
        
        diff = pred_price - current_price
        profit = 0
        
        if diff > threshold: 
            take_profit_price = current_price + tp_val
            stop_loss_price = current_price - sl_val
            
            if actual_next_low <= stop_loss_price:
                profit = (stop_loss_price - current_price - spread_cost) * trade_amount
            elif actual_next_high >= take_profit_price:
                profit = (take_profit_price - current_price - spread_cost) * trade_amount
                wins += 1
            else:
                profit = (actual_next_close - current_price - spread_cost) * trade_amount
                if profit > 0: wins += 1
            total_trades += 1
            
        elif diff < -threshold: 
            take_profit_price = current_price - tp_val
            stop_loss_price = current_price + sl_val
            
            if actual_next_high >= stop_loss_price:
                profit = (current_price - stop_loss_price - spread_cost) * trade_amount
            elif actual_next_low <= take_profit_price:
                profit = (current_price - take_profit_price - spread_cost) * trade_amount
                wins += 1
            else:
                profit = (current_price - actual_next_close - spread_cost) * trade_amount
                if profit > 0: wins += 1
            total_trades += 1
            
        balance += profit
        cumulative_profit.append(balance)
        dates.append(y_test_close.index[i])
        
    return {
        "dates": dates,
        "profits": cumulative_profit,
        "final_balance": balance,
        "total_trades": total_trades,
        "win_rate": (wins / total_trades * 100) if total_trades > 0 else 0
    }

# --- メイン画面構成 ---
tab1, tab2, tab3 = st.tabs(["🔮 未来予測", "📊 バックテスト", "⚙️ 自動最適化 (Quant)"])

df_base = get_historical_data(ticker)

# === タブ1: 未来予測 ===
with tab1:
    st.header(f"🔮 {selected_label} 未来予測")
    
    if df_base is None or df_base.empty:
        st.error(f"データの取得に失敗しました。{ticker} のデータが存在しないか、通信エラーです。")
    else:
        if st.button("最新レートで予測する", type="primary"):
            with st.spinner("AIが思考中..."):
                df_future = df_base.copy()
                realtime = get_realtime_price(ticker)
                if realtime:
                    df_future.iloc[-1, df_future.columns.get_loc('Close')] = realtime
                
                # 特徴量
                df_future['SMA_Short'] = df_future['Close'].rolling(window=p_sma_s).mean()
                df_future['SMA_Long'] = df_future['Close'].rolling(window=p_sma_l).mean()
                df_future['Change'] = df_future['Close'].pct_change()
                df_future['Std'] = df_future['Close'].rolling(window=20).std()
                df_future['Upper_Band'] = df_future['SMA_Long'] + (df_future['Std'] * 2)
                df_future['Lower_Band'] = df_future['SMA_Long'] - (df_future['Std'] * 2)
                
                df_future['Next_Close'] = df_future['Close'].shift(-1)
                
                features = ['Close', 'SMA_Short', 'SMA_Long', 'Change', 'Upper_Band', 'Lower_Band']
                X = df_future[features]
                y = df_future['Next_Close']
                
                if len(X) == 0:
                    st.error("データ不足のため予測できません。")
                else:
                    latest_row = X.iloc[[-1]]
                    X_train = X.iloc[:-1].dropna()
                    y_train = y.iloc[:-1].dropna()
                    
                    if len(X_train) == 0:
                        st.error("有効な学習データがありません。")
                    else:
                        common_idx = X_train.index.intersection(y_train.index)
                        X_train = X_train.loc[common_idx]
                        y_train = y_train.loc[common_idx]
                        
                        latest_row = latest_row.fillna(method='ffill').fillna(0)

                        model = RandomForestRegressor(n_estimators=p_n_est, random_state=42)
                        model.fit(X_train, y_train)
                        
                        pred_price = model.predict(latest_row)[0]
                        current_price = df_future['Close'].iloc[-1]
                        diff = pred_price - current_price
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("現在レート", f"{current_price:.2f} {curr_unit}")
                        with c2:
                            st.metric("AI予測 (Next 1h)", f"{pred_price:.2f} {curr_unit}", delta=f"{diff:.2f} {curr_unit}")
                        
                        st.markdown("---")
                        if diff > p_threshold:
                            st.success(f"📈 **買いシグナル** detected!")
                            st.markdown(f"""
                            - **エントリー**: {current_price:.2f}
                            - **利確目標 (TP)**: {current_price + p_tp:.2f}
                            - **損切ライン (SL)**: {current_price - p_sl:.2f}
                            """)
                        elif diff < -p_threshold:
                            st.error(f"📉 **売りシグナル** detected!")
                            st.markdown(f"""
                            - **エントリー**: {current_price:.2f}
                            - **利確目標 (TP)**: {current_price - p_tp:.2f}
                            - **損切ライン (SL)**: {current_price + p_sl:.2f}
                            """)
                        else:
                            st.warning("✋ 様子見 (予測幅が小さいです)")

                        # グラフ
                        st.subheader("直近チャート")
                        chart_data = df_future.tail(72)
                        fig, ax = plt.subplots(figsize=(12, 5))
                        ax.plot(chart_data.index, chart_data['Close'], label="History", color="gold" if "Gold" in selected_label else "blue")
                        
                        next_time = chart_data.index[-1] + datetime.timedelta(hours=1)
                        ax.scatter([next_time], [pred_price], color="red", s=150, label="AI Prediction", zorder=5, edgecolors='white')
                        ax.plot([chart_data.index[-1], next_time], [current_price, pred_price], color="red", linestyle=":", alpha=0.8)
                        
                        if abs(diff) > p_threshold:
                            if diff > 0: 
                                ax.axhline(y=current_price + p_tp, color='green', linestyle='--', alpha=0.5, label="Take Profit")
                                ax.axhline(y=current_price - p_sl, color='red', linestyle='--', alpha=0.5, label="Stop Loss")
                            else: 
                                ax.axhline(y=current_price - p_tp, color='green', linestyle='--', alpha=0.5, label="Take Profit")
                                ax.axhline(y=current_price + p_sl, color='red', linestyle='--', alpha=0.5, label="Stop Loss")

                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d日 %H:00'))
                        plt.xticks(rotation=45)
                        ax.grid(True, linestyle='--', alpha=0.6)
                        ax.legend()
                        st.pyplot(fig)

# === タブ2: バックテスト ===
with tab2:
    st.header("リスク管理込みのバックテスト")
    st.info(f"銘柄: **{selected_label}** / 取引量: **{trade_amount}** でシミュレーションします。")
    
    p_days = st.slider("検証期間 (日)", 7, 90, 30)

    if st.button("この設定でテスト実行"):
        if df_base is None or df_base.empty:
            st.error("データが空のためバックテストできません。")
        else:
            with st.spinner("シミュレーション中..."):
                # ★修正: trade_amountを渡す
                res = run_backtest_logic(df_base, params, p_days, spread_cost, trade_amount)
                
            if res:
                profit = res['final_balance'] - 1000000
                c1, c2, c3 = st.columns(3)
                c1.metric("純利益 (参考値)", f"{int(profit):,} 円", delta_color="normal" if profit>0 else "inverse")
                c2.metric("取引回数", f"{res['total_trades']} 回")
                c3.metric("勝率", f"{res['win_rate']:.1f} %")
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(res['dates'], res['profits'], label="Total Asset", color="green")
                ax.set_title("Asset Growth")
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("バックテスト結果が計算できませんでした（取引なし、またはデータ不足）。")

# === タブ3: Optuna最適化 ===
with tab3:
    st.header("👑 クオンツ・モード (TP/SL最適化)")
    st.markdown(f"**{selected_label}** に最適な設定をAIに探させます。")
    
    opt_days = st.slider("最適化する検証期間 (日)", 14, 60, 30)
    n_trials = st.slider("試行回数 (Trials)", 10, 100, 20)
    
    if st.button("最強のリスク管理設定を探す", type="primary"):
        if df_base is None or df_base.empty:
            st.error("データが空のため最適化できません。")
        else:
            status = st.empty()
            progress_bar = st.progress(0)
            
            def objective(trial):
                # 銘柄別の探索範囲
                if ticker == "GC=F" or ticker == "XAUUSD=X":
                    # ゴールド: 大きく動く
                    t_th = trial.suggest_float("threshold", 1.00, 10.00)
                    t_tp = trial.suggest_float("tp", 2.00, 30.00)
                    t_sl = trial.suggest_float("sl", 2.00, 20.00)
                elif ticker == "BTC-USD":
                    t_th = trial.suggest_float("threshold", 50.0, 500.0)
                    t_tp = trial.suggest_float("tp", 100.0, 2000.0)
                    t_sl = trial.suggest_float("sl", 100.0, 1000.0)
                else:
                    # ドル円: 小さく動く
                    t_th = trial.suggest_float("threshold", 0.01, 0.15)
                    t_tp = trial.suggest_float("tp", 0.10, 1.00)
                    t_sl = trial.suggest_float("sl", 0.05, 0.50)

                trial_params = {
                    "threshold": t_th,
                    "tp": t_tp,
                    "sl": t_sl,
                    "sma_short": trial.suggest_int("sma_short", 3, 15),
                    "sma_long": trial.suggest_int("sma_long", 20, 60),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 150)
                }
                # ★修正: trade_amountを渡す
                res = run_backtest_logic(df_base, trial_params, opt_days, spread_cost, trade_amount)
                
                if res and res['total_trades'] > 5: 
                    return res['final_balance']
                else:
                    return 0 

            study = optuna.create_study(direction="maximize")
            
            for i in range(n_trials):
                study.optimize(objective, n_trials=1)
                progress = (i + 1) / n_trials
                progress_bar.progress(progress)
                best_profit = study.best_value - 1000000
                status.text(f"試行 {i+1}/{n_trials} 完了... 暫定1位: +{int(best_profit):,}")
            
            st.success("探索完了！")
            best_params = study.best_params
            
            st.subheader("🏆 発見された最適設定")
            c1, c2, c3 = st.columns(3)
            c1.metric("閾値", f"{best_params['threshold']:.4f}")
            c2.metric("利確 TP", f"{best_params['tp']:.4f}")
            c3.metric("損切 SL", f"{best_params['sl']:.4f}")
            
            st.info("👆 サイドバーの設定欄に入力して、バックテストを再実行してください！")
