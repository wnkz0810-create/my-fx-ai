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
st.set_page_config(page_title="FX AI Quant System", layout="wide")

st.title("🤖 FX AI Quant System")
st.write("予測、バックテスト、そして**「数学的な最適解の自動探索（Optuna）」**を備えた統合システムです。")

# --- サイドバー設定 ---
st.sidebar.header("基本設定")
ticker = st.sidebar.text_input("通貨ペア", "JPY=X")

# データ取得（キャッシュ化して高速化）
@st.cache_data(ttl=3600)
def get_historical_data(ticker_symbol, period="2y", interval="1h"):
    df = yf.download(ticker_symbol, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Close']].rename(columns={'Close': 'JPY'})
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

# 特徴量作成とバックテストを行う関数
def run_backtest_logic(df_original, params, test_period_days):
    df = df_original.copy()
    
    # パラメータ展開
    sma_s = params.get('sma_short', 5)
    sma_l = params.get('sma_long', 25)
    threshold = params.get('threshold', 0.05)
    n_estimators = params.get('n_estimators', 100)
    
    # 特徴量作成
    df['SMA_Short'] = df['JPY'].rolling(window=sma_s).mean()
    df['SMA_Long'] = df['JPY'].rolling(window=sma_l).mean()
    df['Change'] = df['JPY'].pct_change()
    
    df['Std'] = df['JPY'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_Long'] + (df['Std'] * 2)
    df['Lower_Band'] = df['SMA_Long'] - (df['Std'] * 2)

    df['Next_Close'] = df['JPY'].shift(-1)
    df = df.dropna()

    features = ['JPY', 'SMA_Short', 'SMA_Long', 'Change', 'Upper_Band', 'Lower_Band']
    X = df[features]
    y = df['Next_Close']

    test_rows = test_period_days * 24
    if len(df) < test_rows + 100:
        return None
        
    X_train = X.iloc[:-test_rows]
    y_train = y.iloc[:-test_rows]
    X_test = X.iloc[-test_rows:]
    y_test = y.iloc[-test_rows:]
    price_test = df['JPY'].iloc[-test_rows:]

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
    
    for i in range(len(X_test)):
        current_price = price_test.iloc[i]
        pred_price = predictions[i]
        actual_next = y_test.iloc[i]
        
        diff = pred_price - current_price
        profit = 0
        
        if diff > threshold: 
            profit = (actual_next - current_price - spread_cost) * trade_amount
            total_trades += 1
            if profit > 0: wins += 1
            
        elif diff < -threshold: 
            profit = (current_price - actual_next - spread_cost) * trade_amount
            total_trades += 1
            if profit > 0: wins += 1
            
        balance += profit
        cumulative_profit.append(balance)
        dates.append(y_test.index[i])
        
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

# 左サイドバーの入力値を取得（全タブで共通利用）
st.sidebar.markdown("---")
st.sidebar.header("パラメータ設定")
st.sidebar.caption("※「自動最適化」で出た数値をここに入力してください")
p_threshold = st.sidebar.number_input("エントリー閾値 (円)", 0.010, 0.200, 0.050, step=0.001, format="%.3f")
p_n_est = st.sidebar.number_input("決定木の数", 10, 300, 100)
p_sma_s = st.sidebar.number_input("短期SMA期間", 2, 20, 5)
p_sma_l = st.sidebar.number_input("長期SMA期間", 20, 100, 25)

params = {
    "threshold": p_threshold,
    "sma_short": p_sma_s,
    "sma_long": p_sma_l,
    "n_estimators": p_n_est
}

# === タブ1: 未来予測 ===
with tab1:
    st.header("🔮 AIによる未来予測")
    st.write("左側のサイドバーで設定したパラメータ（最適値）に基づいて、**次の1時間の動き**を予測します。")
    
    if st.button("最新レートで予測する", type="primary"):
        with st.spinner("AIが思考中..."):
            # 1. データの準備（特徴量作成）
            df_future = df_base.copy()
            
            # リアルタイムレートの上書き
            realtime = get_realtime_price(ticker)
            if realtime:
                df_future.iloc[-1, df_future.columns.get_loc('JPY')] = realtime
            
            # 特徴量計算
            df_future['SMA_Short'] = df_future['JPY'].rolling(window=p_sma_s).mean()
            df_future['SMA_Long'] = df_future['JPY'].rolling(window=p_sma_l).mean()
            df_future['Change'] = df_future['JPY'].pct_change()
            df_future['Std'] = df_future['JPY'].rolling(window=20).std()
            df_future['Upper_Band'] = df_future['SMA_Long'] + (df_future['Std'] * 2)
            df_future['Lower_Band'] = df_future['SMA_Long'] - (df_future['Std'] * 2)
            
            # 「次の足」を作るための正解ラベル作成（学習用）
            df_future['Next_Close'] = df_future['JPY'].shift(-1)
            
            # 特徴量
            features = ['JPY', 'SMA_Short', 'SMA_Long', 'Change', 'Upper_Band', 'Lower_Band']
            X = df_future[features]
            y = df_future['Next_Close']
            
            # 学習データ（最後以外）と、予測したいデータ（最後）
            # dropnaすると最後の行（Next_CloseがNaN）が消えるので、予測用にとっておく
            latest_row = X.iloc[[-1]] # これが「今」
            
            # 学習用データ
            X_train = X.iloc[:-1].dropna()
            y_train = y.iloc[:-1].dropna()
            
            # インデックスを合わせる
            common_idx = X_train.index.intersection(y_train.index)
            X_train = X_train.loc[common_idx]
            y_train = y_train.loc[common_idx]
            
            # 2. モデル学習
            model = RandomForestRegressor(n_estimators=p_n_est, random_state=42)
            model.fit(X_train, y_train)
            
            # 3. 予測
            pred_price = model.predict(latest_row)[0]
            current_price = df_future['JPY'].iloc[-1]
            diff = pred_price - current_price
            
            # 4. 結果表示
            c1, c2 = st.columns(2)
            with c1:
                st.metric("現在レート (Realtime)", f"{current_price:.3f} 円")
            with c2:
                st.metric("AI予測 (Next 1h)", f"{pred_price:.3f} 円", delta=f"{diff:.3f} 円")
            
            # エントリー判定
            st.markdown("---")
            if diff > p_threshold:
                st.success(f"📈 **買いシグナル (LONG)** - 予測上昇幅 (+{diff:.3f}円) が閾値 ({p_threshold}円) を超えました。")
            elif diff < -p_threshold:
                st.error(f"📉 **売りシグナル (SHORT)** - 予測下落幅 ({diff:.3f}円) が閾値 (-{p_threshold}円) を下回りました。")
            else:
                st.warning(f"✋ **様子見 (WAIT)** - 予測変動幅 ({abs(diff):.3f}円) が閾値未満です。スプレッド負けのリスクがあるため取引しません。")
                
            # 5. グラフ描画 (Matplotlib)
            st.subheader("直近チャートと予測ポイント")
            
            # 直近72時間を表示
            chart_data = df_future.tail(72)
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(chart_data.index, chart_data['JPY'], label="History", color="blue")
            
            # ボリンジャーバンド
            ax.plot(chart_data.index, chart_data['Upper_Band'], color="gray", alpha=0.3, linestyle="--")
            ax.plot(chart_data.index, chart_data['Lower_Band'], color="gray", alpha=0.3, linestyle="--")
            ax.fill_between(chart_data.index, chart_data['Upper_Band'], chart_data['Lower_Band'], color='gray', alpha=0.1)
            
            # 予測点
            next_time = chart_data.index[-1] + datetime.timedelta(hours=1)
            ax.scatter([next_time], [pred_price], color="red", s=150, label="AI Prediction", zorder=5, edgecolors='white')
            
            # 現在地点と予測地点を結ぶ
            ax.plot([chart_data.index[-1], next_time], [current_price, pred_price], color="red", linestyle=":", alpha=0.8)
            
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d日 %H:00'))
            plt.xticks(rotation=45)
            ax.set_title("USD/JPY 1H Trend & Prediction")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            
            st.pyplot(fig)


# === タブ2: バックテスト ===
with tab2:
    st.header("パラメータ手動検証")
    st.info("サイドバーで設定した数値を使って、過去の成績を検証します。")
    
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
            ax.set_title("Asset Growth Simulation")
            ax.set_ylabel("JPY")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            st.pyplot(fig)
        else:
            st.error("データ不足またはエラー")

# === タブ3: Optuna最適化 ===
with tab3:
    st.header("👑 クオンツ・モード (パラメータ自動探索)")
    st.markdown("AIが数千通りの組み合わせを高速シミュレーションし、**「今の相場で最も稼げる設定」**を発見します。")
    
    opt_days = st.slider("最適化する検証期間 (日)", 14, 60, 30)
    n_trials = st.slider("試行回数 (Trials)", 10, 100, 20)
    
    if st.button("最強の設定を探す (Start Optimization)", type="primary"):
        status = st.empty()
        progress_bar = st.progress(0)
        
        def objective(trial):
            trial_params = {
                "threshold": trial.suggest_float("threshold", 0.01, 0.15), 
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
            
        st.success("探索完了！最強のパラメータが見つかりました。")
        
        best_params = study.best_params
        best_value = study.best_value - 1000000
        
        st.subheader(f"🏆 発見された最適設定 (利益: +{int(best_value):,}円)")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("閾値 (Threshold)", f"{best_params['threshold']:.3f} 円")
        c2.metric("短期SMA", f"{best_params['sma_short']}")
        c3.metric("長期SMA", f"{best_params['sma_long']}")
        c4.metric("決定木 (Estimators)", f"{best_params['n_estimators']}")
        
        st.info("👆 サイドバーにこの数値を入力して、「未来予測」タブに戻りましょう！")
        
        try:
            from optuna.visualization.matplotlib import plot_param_importances
            fig = plot_param_importances(study)
            st.pyplot(fig)
        except:
            pass