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
st.set_page_config(page_title="FX AI Quant System (TP/SL)", layout="wide")

st.title("🤖 FX AI Quant System (Pro)")
st.write("予測、バックテスト、最適化に加え、**「利確 (TP)・損切 (SL)」**のリスク管理機能を搭載しました。")

# --- サイドバー設定 ---
st.sidebar.header("基本設定")
ticker = st.sidebar.text_input("通貨ペア", "JPY=X")

# データ取得（OHLCすべて取得）
@st.cache_data(ttl=3600)
def get_historical_data(ticker_symbol, period="2y", interval="1h"):
    # 損切判定のためにHighとLowも必要
    df = yf.download(ticker_symbol, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 必要なカラムをリネームして抽出
    df = df[['Open', 'High', 'Low', 'Close']].rename(columns={
        'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close'
    })
    df = df.dropna()
    return df

def get_realtime_price(ticker_symbol):
    try:
        data = yf.download(ticker_symbol, period="1d", interval="1m", progress=False)
        if len(data) > 0:
            return float(data['Close'].iloc[-1])
    except:
        return None
    return None

# バックテストロジック（TP/SL対応版）
def run_backtest_logic(df_original, params, test_period_days):
    df = df_original.copy()
    
    # パラメータ展開
    sma_s = params.get('sma_short', 5)
    sma_l = params.get('sma_long', 25)
    threshold = params.get('threshold', 0.05)
    n_estimators = params.get('n_estimators', 100)
    
    # TP/SL設定（円単位: 0.1 = 10銭）
    tp_val = params.get('tp', 0.50) # デフォルトは広めに（実質時間決済）
    sl_val = params.get('sl', 0.20) # デフォルト20銭で損切
    
    # 特徴量作成
    df['SMA_Short'] = df['Close'].rolling(window=sma_s).mean()
    df['SMA_Long'] = df['Close'].rolling(window=sma_l).mean()
    df['Change'] = df['Close'].pct_change()
    
    df['Std'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_Long'] + (df['Std'] * 2)
    df['Lower_Band'] = df['SMA_Long'] - (df['Std'] * 2)

    df['Next_Close'] = df['Close'].shift(-1)
    # TP/SL判定用に次の足のHigh/Lowも取得
    df['Next_High'] = df['High'].shift(-1)
    df['Next_Low'] = df['Low'].shift(-1)
    
    df = df.dropna()

    features = ['Close', 'SMA_Short', 'SMA_Long', 'Change', 'Upper_Band', 'Lower_Band']
    X = df[features]
    # 正解ラベルは学習用にはCloseを使う
    y = df['Next_Close']

    test_rows = test_period_days * 24
    if len(df) < test_rows + 100:
        return None
        
    X_train = X.iloc[:-test_rows]
    y_train = y.iloc[:-test_rows]
    X_test = X.iloc[-test_rows:]
    
    # テスト用の正解データ群
    y_test_close = df['Next_Close'].iloc[-test_rows:]
    y_test_high = df['Next_High'].iloc[-test_rows:]
    y_test_low = df['Next_Low'].iloc[-test_rows:]
    price_test = df['Close'].iloc[-test_rows:]

    # 学習
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    balance = 1000000 
    cumulative_profit = []
    dates = []
    
    trade_amount = 10000 
    spread_cost = 0.003 

    total_trades = 0
    wins = 0
    
    # 1時間ごとのシミュレーション
    for i in range(len(X_test)):
        current_price = price_test.iloc[i]
        pred_price = predictions[i]
        
        # 実際の次の足の動き
        actual_next_close = y_test_close.iloc[i]
        actual_next_high = y_test_high.iloc[i]
        actual_next_low = y_test_low.iloc[i]
        
        diff = pred_price - current_price
        profit = 0
        
        # --- ロング (買い) の場合 ---
        if diff > threshold: 
            # 目標価格と損切価格を設定
            take_profit_price = current_price + tp_val
            stop_loss_price = current_price - sl_val
            
            # 判定: その1時間の間にSLかTPに刺さったか？
            # ※保守的に「SLが先に刺さる」判定を優先します（安全側評価）
            if actual_next_low <= stop_loss_price:
                # 損切発動
                profit = (stop_loss_price - current_price - spread_cost) * trade_amount
            elif actual_next_high >= take_profit_price:
                # 利確発動
                profit = (take_profit_price - current_price - spread_cost) * trade_amount
                wins += 1
            else:
                # どちらにも刺さらず1時間経過 -> 時間決済
                profit = (actual_next_close - current_price - spread_cost) * trade_amount
                if profit > 0: wins += 1
            
            total_trades += 1
            
        # --- ショート (売り) の場合 ---
        elif diff < -threshold: 
            take_profit_price = current_price - tp_val
            stop_loss_price = current_price + sl_val
            
            if actual_next_high >= stop_loss_price:
                # 損切発動
                profit = (current_price - stop_loss_price - spread_cost) * trade_amount
            elif actual_next_low <= take_profit_price:
                # 利確発動
                profit = (current_price - take_profit_price - spread_cost) * trade_amount
                wins += 1
            else:
                # 時間決済
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

# 左サイドバー設定
st.sidebar.markdown("---")
st.sidebar.header("パラメータ設定")

p_threshold = st.sidebar.number_input("エントリー閾値 (円)", 0.010, 0.200, 0.050, step=0.001, format="%.3f")

# 新機能: TP/SL設定
st.sidebar.subheader("リスク管理")
p_tp = st.sidebar.number_input("利確幅 TP (円)", 0.05, 5.00, 0.50, step=0.05, help="これ以上儲かったら即決済")
p_sl = st.sidebar.number_input("損切幅 SL (円)", 0.05, 5.00, 0.20, step=0.05, help="これ以上損したら即決済")

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

# === タブ1: 未来予測 ===
with tab1:
    st.header("🔮 AIによる未来予測")
    
    if st.button("最新レートで予測する", type="primary"):
        with st.spinner("AIが思考中..."):
            df_future = df_base.copy()
            realtime = get_realtime_price(ticker)
            if realtime:
                # CloseだけでなくOpen/High/Lowも仮置きする（計算用）
                df_future.iloc[-1, df_future.columns.get_loc('Close')] = realtime
            
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
            
            latest_row = X.iloc[[-1]]
            X_train = X.iloc[:-1].dropna()
            y_train = y.iloc[:-1].dropna()
            
            common_idx = X_train.index.intersection(y_train.index)
            X_train = X_train.loc[common_idx]
            y_train = y_train.loc[common_idx]
            
            model = RandomForestRegressor(n_estimators=p_n_est, random_state=42)
            model.fit(X_train, y_train)
            
            pred_price = model.predict(latest_row)[0]
            current_price = df_future['Close'].iloc[-1]
            diff = pred_price - current_price
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("現在レート", f"{current_price:.3f} 円")
            with c2:
                st.metric("AI予測 (Next 1h)", f"{pred_price:.3f} 円", delta=f"{diff:.3f} 円")
            
            st.markdown("---")
            if diff > p_threshold:
                st.success(f"📈 **買いシグナル** detected!")
                st.markdown(f"""
                - **エントリー**: {current_price:.3f}円
                - **利確目標 (TP)**: {current_price + p_tp:.3f}円
                - **損切ライン (SL)**: {current_price - p_sl:.3f}円
                """)
            elif diff < -p_threshold:
                st.error(f"📉 **売りシグナル** detected!")
                st.markdown(f"""
                - **エントリー**: {current_price:.3f}円
                - **利確目標 (TP)**: {current_price - p_tp:.3f}円
                - **損切ライン (SL)**: {current_price + p_sl:.3f}円
                """)
            else:
                st.warning("✋ 様子見 (予測幅が小さいです)")

            # グラフ描画
            st.subheader("直近チャート")
            chart_data = df_future.tail(72)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(chart_data.index, chart_data['Close'], label="History", color="blue")
            
            next_time = chart_data.index[-1] + datetime.timedelta(hours=1)
            ax.scatter([next_time], [pred_price], color="red", s=150, label="AI Prediction", zorder=5, edgecolors='white')
            ax.plot([chart_data.index[-1], next_time], [current_price, pred_price], color="red", linestyle=":", alpha=0.8)
            
            # SL/TPラインの描画（シグナルが出ている場合）
            if abs(diff) > p_threshold:
                if diff > 0: # Long
                    ax.axhline(y=current_price + p_tp, color='green', linestyle='--', alpha=0.5, label="Take Profit")
                    ax.axhline(y=current_price - p_sl, color='red', linestyle='--', alpha=0.5, label="Stop Loss")
                else: # Short
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
    st.info("サイドバーの「利確幅」「損切幅」の設定も反映してシミュレーションします。")
    
    p_days = st.slider("検証期間 (日)", 7, 90, 30)

    if st.button("この設定でテスト実行"):
        with st.spinner("シミュレーション中..."):
            res = run_backtest_logic(df_base, params, p_days)
            
        if res:
            profit = res['final_balance'] - 1000000
            c1, c2, c3 = st.columns(3)
            c1.metric("純利益", f"{int(profit):,} 円", delta_color="normal" if profit>0 else "inverse")
            c2.metric("取引回数", f"{res['total_trades']} 回")
            c3.metric("勝率", f"{res['win_rate']:.1f} %")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(res['dates'], res['profits'], label="Total Asset", color="green")
            ax.set_title("Asset Growth (with TP/SL)")
            ax.set_ylabel("JPY")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            st.pyplot(fig)

# === タブ3: Optuna最適化 ===
with tab3:
    st.header("👑 クオンツ・モード (TP/SL最適化)")
    st.markdown("最適なエントリー閾値だけでなく、**「どこで損切するのが一番稼げるか？」**もAIに探させます。")
    
    opt_days = st.slider("最適化する検証期間 (日)", 14, 60, 30)
    n_trials = st.slider("試行回数 (Trials)", 10, 100, 20)
    
    if st.button("最強のリスク管理設定を探す", type="primary"):
        status = st.empty()
        progress_bar = st.progress(0)
        
        def objective(trial):
            trial_params = {
                # 閾値
                "threshold": trial.suggest_float("threshold", 0.01, 0.15),
                # 損切・利確もAIに決めさせる！
                "tp": trial.suggest_float("tp", 0.10, 1.00), # 10銭〜1円
                "sl": trial.suggest_float("sl", 0.05, 0.50), # 5銭〜50銭
                # モデル設定
                "sma_short": trial.suggest_int("sma_short", 3, 15),
                "sma_long": trial.suggest_int("sma_long", 20, 60),
                "n_estimators": trial.suggest_int("n_estimators", 50, 150)
            }
            res = run_backtest_logic(df_base, trial_params, opt_days)
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
            status.text(f"試行 {i+1}/{n_trials} 完了... 現在の暫定1位: +{int(best_profit):,}円")
            
        st.success("探索完了！")
        
        best_params = study.best_params
        best_value = study.best_value - 1000000
        
        st.subheader(f"🏆 発見された最適設定 (利益: +{int(best_value):,}円)")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("閾値", f"{best_params['threshold']:.3f} 円")
        c2.metric("利確 TP", f"{best_params['tp']:.3f} 円")
        c3.metric("損切 SL", f"{best_params['sl']:.3f} 円")
        
        st.info("👆 サイドバーの「リスク管理」欄に、このTP/SLの値を入力してください！")
        
        try:
            from optuna.visualization.matplotlib import plot_param_importances
            fig = plot_param_importances(study)
            st.pyplot(fig)
        except:
            pass
