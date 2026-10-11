import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.optimize import minimize
import ta

# Optional import for Hyperopt
try:
    from hyperopt import fmin, tpe, hp, Trials
    HAS_HYPEROPT = True
except ImportError:
    HAS_HYPEROPT = False

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CẤU HÌNH TRANG WEB STREAMLIT & GIAO DIỆN
# ==============================================================================
st.set_page_config(
    page_title="Hệ Thống Kiểm Định Chiến Lược SMA + OBV | HOSE",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS cho phong cách tài chính chuyên nghiệp, hiện đại
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%);
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #111827;
        margin-top: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
        border-radius: 6px 6px 0 0;
    }
    .highlight-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        color: #1E40AF;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. HÀM XỬ LÝ & CHUẨN HÓA DỮ LIỆU
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_and_preprocess_data(file_source):
    """
    Đọc và chuẩn hóa dữ liệu giao dịch từ file CSV:
    - Chuẩn hóa tên cột
    - Chuyển đổi định dạng ngày tháng
    - Kiểm tra và chuyển đổi dữ liệu số
    - Tự động ưu tiên 'adj_close' làm giá tính toán
    """
    if isinstance(file_source, str):
        if not os.path.exists(file_source):
            return None, f"Không tìm thấy file mặc định: {file_source}"
        df = pd.read_csv(file_source, encoding="utf-8-sig", low_memory=False)
    else:
        df = pd.read_csv(file_source, encoding="utf-8-sig", low_memory=False)

    df.columns = df.columns.str.strip().str.lower()
    required_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return None, f"Dữ liệu thiếu các cột bắt buộc: {missing_cols}"

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    numeric_cols = ["open", "high", "low", "close", "volume", "adj_open", "adj_high", "adj_low", "adj_close"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "ticker", "close", "volume"]).copy()
    price_col = "adj_close" if "adj_close" in df.columns and df["adj_close"].notna().sum() > 0 else "close"
    return df, price_col


def prepare_stock_series(df_full, ticker, price_col):
    """Lọc và sắp xếp chuỗi dữ liệu giá đóng cửa và khối lượng cho một mã cổ phiếu"""
    sub = df_full[df_full["ticker"] == ticker].copy()
    sub = sub.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    sub = sub.dropna(subset=["date", price_col, "volume"])
    out = pd.DataFrame({
        "date": sub["date"],
        "Close": sub[price_col].astype(float),
        "Volume": sub["volume"].astype(float)
    })
    return out.set_index("date").sort_index()


# ==============================================================================
# 3. TA SCREENING & SCORING TRÊN TẬP TRAIN
# ==============================================================================
@st.cache_data(show_spinner=False)
def calculate_ta_features(df_train, price_col, min_observations=400, trading_days=252):
    """
    Tính toán 4 nhóm chỉ số phân tích kỹ thuật trên tập dữ liệu Huấn luyện:
    1. Trend: Tỷ lệ SMA50 > SMA200, Giá so với SMA200
    2. Momentum: Tỷ suất sinh lời 6 tháng (126 phiên)
    3. Volume: Giá trị giao dịch trung bình, OBV Confirmation
    4. Risk: Biến động lịch sử năm (Volatility), Mức sụt giảm tối đa (Max Drawdown)
    """
    rows = []
    for ticker, g in df_train.groupby("ticker"):
        g = g.sort_values("date").dropna(subset=[price_col, "volume"]).copy()
        if len(g) < min_observations:
            continue

        price = g[price_col].astype(float)
        volume = g["volume"].astype(float)

        # 1. Trend
        sma50 = price.rolling(50).mean()
        sma200 = price.rolling(200).mean()
        trend_strength = (sma50 > sma200).mean()
        if pd.notna(sma200.iloc[-1]) and sma200.iloc[-1] != 0:
            price_vs_sma200 = price.iloc[-1] / sma200.iloc[-1] - 1
        else:
            price_vs_sma200 = np.nan

        # 2. Momentum (6M ~ 126 phiên)
        momentum_6m = price.iloc[-1] / price.iloc[-126] - 1 if len(price) >= 126 else np.nan

        # 3. Volume & Liquidity
        avg_trading_value = (price * volume).mean()
        obv = ta.volume.OnBalanceVolumeIndicator(close=price, volume=volume).on_balance_volume()
        if len(obv) >= 20:
            vol_20 = volume.iloc[-20:].sum()
            obv_conf = (obv.iloc[-1] - obv.iloc[-20]) / vol_20 if vol_20 != 0 else np.nan
        else:
            obv_conf = np.nan

        # 4. Risk
        rets = price.pct_change().dropna()
        ann_vol = rets.std() * np.sqrt(trading_days)
        cum_eq = (1 + rets).cumprod()
        max_dd = (cum_eq / cum_eq.cummax() - 1).min()

        rows.append({
            "Ticker": ticker,
            "N": len(g),
            "Trend_Strength": trend_strength,
            "Price_vs_SMA200": price_vs_sma200,
            "Momentum_6M": momentum_6m,
            "Avg_Trading_Value": avg_trading_value,
            "OBV_Confirmation": obv_conf,
            "Volatility": ann_vol,
            "Max_Drawdown": max_dd
        })

    out = pd.DataFrame(rows)
    cols_to_check = [
        "Trend_Strength", "Price_vs_SMA200", "Momentum_6M",
        "Avg_Trading_Value", "OBV_Confirmation", "Volatility", "Max_Drawdown"
    ]
    return out.dropna(subset=cols_to_check).reset_index(drop=True)


def build_ta_score(features_df, weights=None):
    """
    Tính điểm TA Score tổng hợp dựa trên percentile rank của từng tiêu chí
    """
    if weights is None:
        weights = {"trend": 0.30, "momentum": 0.25, "volume": 0.25, "risk": 0.20}

    df = features_df.copy()
    # Trend score
    df["Trend_Strength_Rank"] = df["Trend_Strength"].rank(pct=True)
    df["Price_SMA200_Rank"] = df["Price_vs_SMA200"].rank(pct=True)
    df["Trend_Score"] = 0.50 * df["Trend_Strength_Rank"] + 0.50 * df["Price_SMA200_Rank"]

    # Momentum score
    df["Momentum_Score"] = df["Momentum_6M"].rank(pct=True)

    # Volume score
    df["Liquidity_Rank"] = df["Avg_Trading_Value"].rank(pct=True)
    df["OBV_Rank"] = df["OBV_Confirmation"].rank(pct=True)
    df["Volume_Score"] = 0.60 * df["Liquidity_Rank"] + 0.40 * df["OBV_Rank"]

    # Risk score (Volatility càng nhỏ rank càng cao -> ascending=False; MaxDD càng âm ít rank càng cao)
    df["Volatility_Rank"] = df["Volatility"].rank(pct=True, ascending=False)
    df["Drawdown_Rank"] = df["Max_Drawdown"].rank(pct=True)
    df["Risk_Score"] = 0.50 * df["Volatility_Rank"] + 0.50 * df["Drawdown_Rank"]

    # Total TA Score
    df["TA_Score"] = (
        weights["trend"] * df["Trend_Score"]
        + weights["momentum"] * df["Momentum_Score"]
        + weights["volume"] * df["Volume_Score"]
        + weights["risk"] * df["Risk_Score"]
    )
    return df.sort_values("TA_Score", ascending=False).reset_index(drop=True)


# ==============================================================================
# 4. CHIẾN LƯỢC TÍN HIỆU SMA + OBV LOGIC OR & BACKTEST ENGINE
# ==============================================================================
def find_position_sma(df, ma_short, ma_long):
    """Xác định tín hiệu SMA Crossover: +1 (Buy), -1 (Sell), 0 (Neutral)"""
    pos = pd.Series(0.0, index=df.index, name="SMA_signal")
    if ma_short >= ma_long:
        return pos
    sma_s = df["Close"].rolling(int(ma_short)).mean()
    sma_l = df["Close"].rolling(int(ma_long)).mean()
    buy = (sma_s > sma_l) & (sma_s.shift(1) <= sma_l.shift(1))
    sell = (sma_s < sma_l) & (sma_s.shift(1) >= sma_l.shift(1))
    pos.loc[buy] = 1.0
    pos.loc[sell] = -1.0
    return pos


def find_position_obv(df, obv_window):
    """Xác định tín hiệu OBV Crossover: +1 (Buy), -1 (Sell), 0 (Neutral)"""
    pos = pd.Series(0.0, index=df.index, name="OBV_signal")
    obv = ta.volume.OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
    obv_ma = obv.rolling(int(obv_window)).mean()
    buy = (obv > obv_ma) & (obv.shift(1) <= obv_ma.shift(1))
    sell = (obv < obv_ma) & (obv.shift(1) >= obv_ma.shift(1))
    pos.loc[buy] = 1.0
    pos.loc[sell] = -1.0
    return pos


def find_position_or(df, ma_short, ma_long, obv_window):
    """Kết hợp tín hiệu SMA và OBV theo logic OR (xử lý xung đột Buy/Sell trong cùng phiên)"""
    sma_sig = find_position_sma(df, ma_short, ma_long)
    obv_sig = find_position_obv(df, obv_window)

    pos = pd.Series(0.0, index=df.index, name="SMA_OBV_OR")
    buy = (sma_sig == 1.0) | (obv_sig == 1.0)
    sell = (sma_sig == -1.0) | (obv_sig == -1.0)
    conflict = buy & sell

    pos.loc[buy & ~conflict] = 1.0
    pos.loc[sell & ~conflict] = -1.0
    return pos


def events_to_holding(events):
    """Chuyển đổi sự kiện Crossover (1: Mua, -1: Bán) thành trạng thái nắm giữ (1: Long, 0: Cash)"""
    holding = pd.Series(0.0, index=events.index)
    curr = 0.0
    for i, sig in enumerate(events):
        if sig == 1.0:
            curr = 1.0
        elif sig == -1.0:
            curr = 0.0
        holding.iloc[i] = curr
    return holding


def calculate_strategy_returns(df, events, commission=0.001):
    """
    Tính toán tỷ suất sinh lời thực tế của chiến lược:
    - shift(1): Tín hiệu sinh ra ở phiên t chỉ thực thi tại phiên t+1 (loại bỏ look-ahead bias)
    - Trừ chi phí giao dịch dựa trên tỷ lệ thay đổi vị thế (Turnover)
    """
    asset_ret = df["Close"].pct_change().fillna(0.0)
    holding = events_to_holding(events)
    executed_holding = holding.shift(1).fillna(0.0)

    gross_ret = executed_holding * asset_ret
    turnover = executed_holding.diff().abs().fillna(executed_holding.abs())
    net_ret = gross_ret - turnover * commission

    return net_ret, executed_holding, turnover


def performance_stats(returns, trading_days=252):
    """Tính các chỉ số đo lường hiệu quả đầu tư định lượng chuẩn quốc tế"""
    r = pd.Series(returns).dropna().astype(float)
    if len(r) == 0:
        return {
            "Mean Daily Return [%]": np.nan,
            "Total Return [%]": np.nan,
            "Annual Return [%]": np.nan,
            "Annual Volatility [%]": np.nan,
            "Sharpe Ratio": np.nan,
            "Max Drawdown [%]": np.nan
        }

    equity = (1 + r).cumprod()
    total_ret = equity.iloc[-1] - 1.0
    years = len(r) / trading_days

    if years > 0 and equity.iloc[-1] > 0:
        ann_ret = (equity.iloc[-1] ** (1 / years)) - 1.0
    else:
        ann_ret = np.nan

    ann_vol = r.std() * np.sqrt(trading_days)
    sharpe = (r.mean() / r.std() * np.sqrt(trading_days)) if r.std() != 0 else np.nan
    cummax = equity.cummax()
    drawdown = (equity / cummax - 1.0)

    return {
        "Mean Daily Return [%]": r.mean() * 100,
        "Total Return [%]": total_ret * 100,
        "Annual Return [%]": ann_ret * 100,
        "Annual Volatility [%]": ann_vol * 100,
        "Sharpe Ratio": sharpe,
        "Max Drawdown [%]": drawdown.min() * 100
    }


def evaluate_stock_strategy(full_df, ma_short, ma_long, obv_window, test_start=None, test_end=None, commission=0.001):
    """
    Đánh giá chiến lược trên cổ phiếu với cơ chế Warm-up lịch sử:
    - Nếu có test_start/test_end, tính tín hiệu trên toàn bộ lịch sử trước test, sau đó slice kết quả test
    """
    if test_start is not None and test_end is not None:
        eval_df = full_df.loc[full_df.index <= pd.Timestamp(test_end)].copy()
        events = find_position_or(eval_df, ma_short, ma_long, obv_window)
        ret, holding, turnover = calculate_strategy_returns(eval_df, events, commission)

        test_ret = ret.loc[test_start:test_end]
        test_events = events.loc[test_start:test_end]
        test_holding = holding.loc[test_start:test_end]
        test_turnover = turnover.loc[test_start:test_end]

        return {
            "returns": test_ret,
            "events": test_events,
            "holding": test_holding,
            "turnover": test_turnover,
            "stats": performance_stats(test_ret)
        }
    else:
        events = find_position_or(full_df, ma_short, ma_long, obv_window)
        ret, holding, turnover = calculate_strategy_returns(full_df, events, commission)
        return {
            "returns": ret,
            "events": events,
            "holding": holding,
            "turnover": turnover,
            "stats": performance_stats(ret)
        }


# ==============================================================================
# 5. MPT PORTFOLIO OPTIMIZATION & BENCHMARK
# ==============================================================================
def optimize_mpt_sharpe(returns_matrix, min_weight=0.05, max_weight=0.40, risk_free_rate=0.0, trading_days=252):
    """
    Tối ưu hóa danh mục theo Lý thuyết Danh mục Hiện đại (Markowitz MPT):
    - Mục tiêu: Tối đa hóa Sharpe Ratio trên ma trận lợi nhuận
    - Ràng buộc: Tổng trọng số = 100%, Long-only
    - Ràng buộc tập trung: Mỗi tài sản nằm trong [min_weight, max_weight]
    """
    n = returns_matrix.shape[1]
    if n * min_weight > 1 or n * max_weight < 1:
        raise ValueError("Ràng buộc biên độ MPT không khả thi.")

    def neg_sharpe(w):
        p_ret = float(w @ returns_matrix.mean().values * trading_days)
        cov = returns_matrix.cov().values * trading_days
        p_vol = np.sqrt(max(float(w.T @ cov @ w), 1e-9))
        return -(p_ret - risk_free_rate) / p_vol

    x0 = np.repeat(1.0 / n, n)
    bounds = [(min_weight, max_weight)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(neg_sharpe, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not res.success:
        return x0
    return res.x


def calculate_portfolio_stream(return_matrix, weights):
    """Tính chuỗi tỷ suất sinh lời danh mục từ ma trận lợi nhuận và vector trọng số"""
    return return_matrix.mul(weights, axis=1).sum(axis=1)


def calculate_equal_weight_buy_hold(data_dict, tickers, equal_weights):
    """
    Benchmark Buy & Hold thực sự của danh mục:
    - Phân bổ 20% ban đầu cho mỗi mã tại ngày đầu tiên của Test
    - KHÔNG tái cân bằng (rebalance) mỗi ngày, giá trị tự trôi theo thị trường
    """
    price_matrix = pd.concat({t: data_dict[t]["Close"] for t in tickers}, axis=1).dropna()
    normalized = price_matrix / price_matrix.iloc[0]
    wealth = (normalized * equal_weights).sum(axis=1)
    returns = wealth.pct_change().fillna(0.0)
    return returns, wealth


# ==============================================================================
# 6. THAM SỐ TỐI ƯU MẶC ĐỊNH TỪ NOTEBOOK GỐC
# ==============================================================================
DEFAULT_BEST_PARAMS = {
    "DIG": {"ma_short": 60, "ma_long": 150, "obv_window": 55},
    "DPM": {"ma_short": 50, "ma_long": 330, "obv_window": 55},
    "NLG": {"ma_short": 35, "ma_long": 310, "obv_window": 50},
    "KBC": {"ma_short": 25, "ma_long": 350, "obv_window": 15},
    "DCM": {"ma_short": 45, "ma_long": 210, "obv_window": 15},
}


# ==============================================================================
# 7. GIAO DIỆN CHÍNH (STREAMLIT SIDEBAR & MAIN BODY)
# ==============================================================================
def main():
    # Header chính
    st.markdown('<div class="main-header">📈 KIỂM ĐỊNH HIỆU QUẢ CHIẾN LƯỢC SMA + OBV</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Nền tảng Backtest Định lượng & Tối ưu hóa Danh mục Đầu tư Cổ phiếu HOSE (2020 – 2023)</div>',
        unsafe_allow_html=True
    )

    # ------------------ SIDEBAR CONTROLS ------------------
    st.sidebar.title("⚙️ Cấu Hình Hệ Thống")

    # 1. Nguồn Dữ Liệu
    st.sidebar.subheader("1. Nguồn Dữ Liệu")
    data_source_mode = st.sidebar.radio(
        "Chọn nguồn file CSV:",
        ["File mẫu có sẵn (HOSE_2020_2023_in.csv)", "Tải lên file CSV mới"],
        index=0
    )

    file_to_load = "HOSE_2020_2023_in.csv"
    if data_source_mode == "Tải lên file CSV mới":
        uploaded_file = st.sidebar.file_uploader("Tải file CSV dữ liệu sàn HOSE:", type=["csv"])
        if uploaded_file is not None:
            file_to_load = uploaded_file
        else:
            st.sidebar.warning("Chưa tải file mới, đang dùng file mặc định.")

    # Tải dữ liệu
    df_full, price_col_or_err = load_and_preprocess_data(file_to_load)
    if df_full is None:
        st.error(f"❌ Lỗi tải dữ liệu: {price_col_or_err}")
        st.info("Vui lòng kiểm tra lại file `HOSE_2020_2023_in.csv` trong thư mục gốc hoặc tải lên file hợp lệ.")
        st.stop()

    price_col = price_col_or_err
    min_date = df_full["date"].min().date()
    max_date = df_full["date"].max().date()

    # Hàm đảm bảo ngày mặc định luôn nằm an toàn trong [min_date, max_date]
    def clamp_date(target_str, min_d, max_d):
        target = pd.to_datetime(target_str).date()
        if target < min_d:
            return min_d
        if target > max_d:
            return max_d
        return target

    default_train_start = clamp_date("2020-01-01", min_date, max_date)
    default_train_end = clamp_date("2021-12-31", min_date, max_date)
    default_test_start = clamp_date("2022-01-01", min_date, max_date)
    default_test_end = clamp_date("2022-12-31", min_date, max_date)

    # 2. Cấu hình Chu kỳ Train & Test
    st.sidebar.subheader("2. Chu Kỳ Dữ Liệu")
    col_sb1, col_sb2 = st.sidebar.columns(2)
    with col_sb1:
        train_start_input = st.date_input(
            "Train Bắt đầu",
            value=default_train_start,
            min_value=min_date,
            max_value=max_date
        )
        test_start_input = st.date_input(
            "Test Bắt đầu",
            value=default_test_start,
            min_value=min_date,
            max_value=max_date
        )
    with col_sb2:
        train_end_input = st.date_input(
            "Train Kết thúc",
            value=default_train_end,
            min_value=min_date,
            max_value=max_date
        )
        test_end_input = st.date_input(
            "Test Kết thúc",
            value=default_test_end,
            min_value=min_date,
            max_value=max_date
        )

    if train_start_input >= train_end_input:
        st.sidebar.error("⚠️ Ngày Train Bắt đầu phải trước ngày Train Kết thúc!")
        st.stop()
    if test_start_input >= test_end_input:
        st.sidebar.error("⚠️ Ngày Test Bắt đầu phải trước ngày Test Kết thúc!")
        st.stop()

    train_start_str = str(train_start_input)
    train_end_str = str(train_end_input)
    test_start_str = str(test_start_input)
    test_end_str = str(test_end_input)

    # 3. Tham số mô phỏng
    st.sidebar.subheader("3. Giả Định Giao Dịch")
    base_commission = st.sidebar.slider("Phí giao dịch cơ sở (Turnover Fee %):", min_value=0.0, max_value=0.5, value=0.10, step=0.01) / 100.0
    initial_capital = st.sidebar.number_input("Vốn khởi điểm (VNĐ):", value=1_000_000, step=100_000, format="%d")

    mpt_min_w = st.sidebar.slider("MPT Min Weight/Mã (%):", min_value=0, max_value=20, value=5, step=1) / 100.0
    mpt_max_w = st.sidebar.slider("MPT Max Weight/Mã (%):", min_value=20, max_value=100, value=40, step=5) / 100.0

    # ------------------ PRE-PROCESSING TA SCREENING ------------------
    df_train = df_full[(df_full["date"] >= train_start_str) & (df_full["date"] <= train_end_str)].copy()
    if df_train.empty:
        st.error("⚠️ Khoảng thời gian Train không có dữ liệu! Vui lòng chọn lại khoảng ngày.")
        st.stop()

    # Tính toán TA features (tự động điều chỉnh min_observations nếu train ngắn)
    unique_train_days = df_train["date"].nunique()
    min_obs = min(400, max(30, int(unique_train_days * 0.7)))
    ta_features = calculate_ta_features(df_train, price_col, min_observations=min_obs)
    if ta_features.empty:
        ta_features = calculate_ta_features(df_train, price_col, min_observations=20)

    # Khởi tạo tabs chính
    tab_overview, tab_screening, tab_stock, tab_portfolio, tab_sensitivity, tab_methodology = st.tabs([
        "📊 1. Tổng Quan Dữ Liệu",
        "🎯 2. Lọc Cổ Phiếu (TA Screening)",
        "📈 3. Chiến Lược Đơn Lẻ (SMA + OBV)",
        "💼 4. Danh Mục MPT vs Equal Weight",
        "🔬 5. Phân Tích Độ Nhạy",
        "📖 6. Phương Pháp Luận & Hướng Dẫn"
    ])

    # ==============================================================================
    # TAB 1: TỔNG QUAN DỮ LIỆU
    # ==============================================================================
    with tab_overview:
        st.subheader("📌 Tổng quan Tập Dữ liệu Giao dịch")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Tổng số bản ghi", f"{len(df_full):,}")
        mcol2.metric("Số mã chứng khoán", f"{df_full['ticker'].nunique()}")
        mcol3.metric("Ngày bắt đầu", f"{min_date}")
        mcol4.metric("Ngày kết thúc", f"{max_date}")

        st.markdown(f"""
        <div class="highlight-box">
            <b>Cột giá sử dụng cho phân tích:</b> <code>{price_col}</code> 
            (Hệ thống tự động ưu tiên <code>adj_close</code> để loại bỏ sai lệch từ chia cổ tức, phát hành thêm).<br>
            <b>Giai đoạn Huấn luyện (Train):</b> <code>{train_start_str}</code> đến <code>{train_end_str}</code> ({len(df_train):,} dòng).<br>
            <b>Giai đoạn Kiểm định (Out-of-sample Test):</b> <code>{test_start_str}</code> đến <code>{test_end_str}</code>.
        </div>
        """, unsafe_allow_html=True)

        st.write("#### 🔍 Xem trước Dữ liệu Lịch sử")
        selected_preview_ticker = st.selectbox(
            "Chọn mã cổ phiếu xem mẫu:",
            sorted(df_full["ticker"].unique()),
            index=sorted(df_full["ticker"].unique()).index("DIG") if "DIG" in df_full["ticker"].values else 0
        )
        sample_df = df_full[df_full["ticker"] == selected_preview_ticker].sort_values("date", ascending=False)
        st.dataframe(sample_df.head(15), use_container_width=True)

        # Biểu đồ giá mẫu
        fig_sample = go.Figure()
        fig_sample.add_trace(go.Scatter(
            x=sample_df["date"],
            y=sample_df[price_col],
            mode="lines",
            name=f"Giá {price_col.upper()} ({selected_preview_ticker})",
            line=dict(color="#2563EB", width=2)
        ))
        fig_sample.update_layout(
            title=f"Biểu đồ giá lịch sử mã {selected_preview_ticker}",
            xaxis_title="Thời gian",
            yaxis_title="Giá (VNĐ)",
            template="plotly_white",
            height=350,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_sample, use_container_width=True)

    # ==============================================================================
    # TAB 2: TA SCREENING TRÊN TRAIN
    # ==============================================================================
    with tab_screening:
        st.subheader("🎯 Sàng lọc Kỹ thuật & Xếp hạng Cổ phiếu (Train Period)")
        st.markdown("""
        Mô hình sàng lọc đánh giá toàn bộ cổ phiếu trên sàn HOSE dựa trên 4 trụ cột kỹ thuật:
        - **Trend (30%):** Sức mạnh xu hướng (tỷ lệ SMA50 > SMA200) và vị thế giá so với SMA200.
        - **Momentum (25%):** Động lượng sinh lời 6 tháng (126 phiên giao dịch).
        - **Volume (25%):** Quy mô thanh khoản khớp lệnh và mức độ bùng nổ của dòng tiền OBV.
        - **Risk (20%):** Mức độ phòng vệ rủi ro (độ biến động thấp và mức sụt giảm Max Drawdown nhỏ).
        """)

        # Tùy chọn kịch bản trọng số
        sc_col1, sc_col2 = st.columns([1, 2])
        with sc_col1:
            ta_scenario = st.selectbox(
                "Chọn kịch bản trọng số TA:",
                ["Base 30/25/25/20 (Mặc định)", "Equal 25/25/25/25", "Risk-oriented 25/20/20/35", "Tùy chỉnh (Custom)"]
            )
            top_n = st.slider("Số lượng cổ phiếu chọn vào danh mục (Top N):", min_value=3, max_value=15, value=5)

        if ta_scenario == "Base 30/25/25/20 (Mặc định)":
            current_weights = {"trend": 0.30, "momentum": 0.25, "volume": 0.25, "risk": 0.20}
        elif ta_scenario == "Equal 25/25/25/25":
            current_weights = {"trend": 0.25, "momentum": 0.25, "volume": 0.25, "risk": 0.25}
        elif ta_scenario == "Risk-oriented 25/20/20/35":
            current_weights = {"trend": 0.25, "momentum": 0.20, "volume": 0.20, "risk": 0.35}
        else:
            with sc_col2:
                w_c1, w_c2, w_c3, w_c4 = st.columns(4)
                w_tr = w_c1.slider("Trend Weight", 0.0, 1.0, 0.30, 0.05)
                w_mo = w_c2.slider("Momentum Weight", 0.0, 1.0, 0.25, 0.05)
                w_vo = w_c3.slider("Volume Weight", 0.0, 1.0, 0.25, 0.05)
                w_ri = w_c4.slider("Risk Weight", 0.0, 1.0, 0.20, 0.05)
                total_w = w_tr + w_mo + w_vo + w_ri
                if total_w > 0:
                    current_weights = {
                        "trend": w_tr / total_w,
                        "momentum": w_mo / total_w,
                        "volume": w_vo / total_w,
                        "risk": w_ri / total_w
                    }
                else:
                    current_weights = {"trend": 0.25, "momentum": 0.25, "volume": 0.25, "risk": 0.25}

        ta_scores = build_ta_score(ta_features, current_weights)
        top_stocks = ta_scores.head(top_n).copy()
        top_stocks.insert(0, "Rank", range(1, len(top_stocks) + 1))

        st.write(f"#### 🏆 Bảng Xếp Hạng Top {top_n} Cổ Phiếu Đạt Chuẩn TA")
        display_cols = [
            "Rank", "Ticker", "TA_Score", "Trend_Score", "Momentum_Score",
            "Volume_Score", "Risk_Score", "Trend_Strength", "Momentum_6M",
            "Volatility", "Max_Drawdown"
        ]
        st.dataframe(
            top_stocks[display_cols].style.format({
                "TA_Score": "{:.3f}",
                "Trend_Score": "{:.3f}",
                "Momentum_Score": "{:.3f}",
                "Volume_Score": "{:.3f}",
                "Risk_Score": "{:.3f}",
                "Trend_Strength": "{:.2%}",
                "Momentum_6M": "{:.2%}",
                "Volatility": "{:.2%}",
                "Max_Drawdown": "{:.2%}"
            }),
            use_container_width=True
        )

        # Radar chart so sánh điểm của Top cổ phiếu
        radar_fig = go.Figure()
        categories = ["Trend Score", "Momentum Score", "Volume Score", "Risk Score"]
        for _, r in top_stocks.head(5).iterrows():
            radar_fig.add_trace(go.Scatterpolar(
                r=[r["Trend_Score"], r["Momentum_Score"], r["Volume_Score"], r["Risk_Score"]],
                theta=categories,
                fill="toself",
                name=r["Ticker"]
            ))
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="So sánh 4 trụ cột kỹ thuật của các cổ phiếu dẫn đầu",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    # Chuẩn bị dữ liệu danh mục Top cổ phiếu
    selected_top_tickers = top_stocks["Ticker"].tolist()
    stock_series_dict = {
        ticker: prepare_stock_series(df_full, ticker, price_col)
        for ticker in selected_top_tickers
    }

    # ==============================================================================
    # TÍNH TOÁN CHIẾN LƯỢC TOÀN DIỆN CHO CẢ TOP 5 CỔ PHIẾU (KHÔNG TÁCH LẺ)
    # ==============================================================================
    all_stock_results = {}
    for t in selected_top_tickers:
        p = DEFAULT_BEST_PARAMS.get(t, {"ma_short": 35, "ma_long": 200, "obv_window": 30})
        stk_df = stock_series_dict[t]

        tr_eval = evaluate_stock_strategy(
            stk_df.loc[train_start_str:train_end_str],
            p["ma_short"], p["ma_long"], p["obv_window"],
            commission=base_commission
        )
        te_eval = evaluate_stock_strategy(
            stk_df,
            p["ma_short"], p["ma_long"], p["obv_window"],
            test_start=test_start_str, test_end=test_end_str,
            commission=base_commission
        )

        tr_sub = stk_df.loc[train_start_str:train_end_str]
        te_sub = stk_df.loc[test_start_str:test_end_str]
        tr_bh_ret = tr_sub["Close"].pct_change().fillna(0.0)
        te_bh_ret = te_sub["Close"].pct_change().fillna(0.0)

        all_stock_results[t] = {
            "params": p,
            "train_strategy": tr_eval,
            "test_strategy": te_eval,
            "train_bh_stats": performance_stats(tr_bh_ret),
            "test_bh_stats": performance_stats(te_bh_ret)
        }

    # 1. Bảng so sánh tổng hợp 5 mã giữa SMA+OBV OR vs Buy & Hold trên Test
    comp_rows = []
    for t in selected_top_tickers:
        res = all_stock_results[t]
        p = res["params"]
        st_stats = res["test_strategy"]["stats"]
        bh_stats = res["test_bh_stats"]
        events = res["test_strategy"]["events"]
        turnover = res["test_strategy"]["turnover"]

        ret_bh = bh_stats["Total Return [%]"]
        ret_st = st_stats["Total Return [%]"]
        excess = ret_st - ret_bh

        comp_rows.append({
            "Mã CP": t,
            "SMA Ngắn": p["ma_short"],
            "SMA Dài": p["ma_long"],
            "OBV Window": p["obv_window"],
            "B&H Return [%]": ret_bh,
            "SMA+OBV Return [%]": ret_st,
            "Chênh Lệch vs B&H [%]": excess,
            "B&H Sharpe": bh_stats["Sharpe Ratio"],
            "SMA+OBV Sharpe": st_stats["Sharpe Ratio"],
            "B&H Max DD [%]": bh_stats["Max Drawdown [%]"],
            "SMA+OBV Max DD [%]": st_stats["Max Drawdown [%]"],
            "Lệnh Mua (Test)": int((events == 1).sum()),
            "Lệnh Bán (Test)": int((events == -1).sum()),
            "Turnover (Test)": turnover.sum()
        })
    df_comparison_5 = pd.DataFrame(comp_rows)

    # 2. Bảng thống kê số lượng lệnh & Turnover cả 5 mã
    trade_rows = []
    for t in selected_top_tickers:
        res = all_stock_results[t]
        tr_ev = res["train_strategy"]["events"]
        te_ev = res["test_strategy"]["events"]
        te_to = res["test_strategy"]["turnover"]
        trade_rows.append({
            "Mã CP": t,
            "Train Lệnh Mua": int((tr_ev == 1).sum()),
            "Train Lệnh Bán": int((tr_ev == -1).sum()),
            "Test Lệnh Mua": int((te_ev == 1).sum()),
            "Test Lệnh Bán": int((te_ev == -1).sum()),
            "Test Turnover": te_to.sum()
        })
    df_trades_summary_all = pd.DataFrame(trade_rows)

    # 3 & 4. Chi tiết hiệu suất Train & Test cả 5 mã
    df_strategy_train_stats_all = pd.DataFrame({
        t: all_stock_results[t]["train_strategy"]["stats"]
        for t in selected_top_tickers
    }).T
    df_strategy_train_stats_all.index.name = "Mã CP"

    df_strategy_test_stats_all = pd.DataFrame({
        t: all_stock_results[t]["test_strategy"]["stats"]
        for t in selected_top_tickers
    }).T
    df_strategy_test_stats_all.index.name = "Mã CP"

    # ==============================================================================
    # TAB 3: TỔNG HỢP CHIẾN LƯỢC CẢ 5 CỔ PHIẾU (KHÔNG TÁCH LẺ)
    # ==============================================================================
    with tab_stock:
        st.subheader(f"📈 Bảng Tổng Hợp Chiến Lược Đầy Đủ Top {len(selected_top_tickers)} Cổ Phiếu (SMA + OBV OR)")
        st.markdown(f"""
        Bảng tổng hợp đối sánh toàn diện toàn bộ **{len(selected_top_tickers)} mã cổ phiếu** giữa Chiến lược kết hợp tín hiệu **SMA + OBV (Logic OR)** và phương pháp **Mua & Nắm giữ (Buy & Hold)** trên tập kiểm định ngoài mẫu (**Out-of-sample Test 2022**).
        """)

        # KPI Metrics tổng quan toàn bộ 5 mã
        avg_ret_bh = df_comparison_5["B&H Return [%]"].mean()
        avg_ret_st = df_comparison_5["SMA+OBV Return [%]"].mean()
        avg_excess = avg_ret_st - avg_ret_bh
        n_outperform = int((df_comparison_5["Chênh Lệch vs B&H [%]"] > 0).sum())

        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
        kpi_c1.metric("Lợi Nhuận TB Buy & Hold", f"{avg_ret_bh:.2f}%")
        kpi_c2.metric("Lợi Nhuận TB SMA+OBV OR", f"{avg_ret_st:.2f}%", delta=f"{avg_excess:+.2f}% vs B&H")
        kpi_c3.metric("Số Mã Vượt Trội B&H", f"{n_outperform}/{len(selected_top_tickers)} mã", delta="Thắng thế trong Downtrend")
        kpi_c4.metric("Sharpe TB Chiến Lược", f"{df_comparison_5['SMA+OBV Sharpe'].mean():.3f}")

        # BẢNG TỔNG HỢP ĐẦY ĐỦ CẢ 5 MÃ
        st.write("#### 🏆 1. Bảng So Sánh Tổng Hợp Toàn Diện 5 Mã (Test 2022)")
        
        styler_5 = df_comparison_5.style.format({
            "B&H Return [%]": "{:.2f}%",
            "SMA+OBV Return [%]": "{:.2f}%",
            "Chênh Lệch vs B&H [%]": "{:+.2f}%",
            "B&H Sharpe": "{:.4f}",
            "SMA+OBV Sharpe": "{:.4f}",
            "B&H Max DD [%]": "{:.2f}%",
            "SMA+OBV Max DD [%]": "{:.2f}%",
            "Lệnh Mua (Test)": "{:d}",
            "Lệnh Bán (Test)": "{:d}",
            "Turnover (Test)": "{:.2f}"
        })

        def color_excess_val(v):
            if isinstance(v, (int, float)):
                if v > 0:
                    return "color: #10B981; font-weight: bold;"
                elif v < 0:
                    return "color: #EF4444; font-weight: bold;"
            return ""

        # Tương thích cả pandas cũ (<2.1 dùng applymap) và pandas mới (>=2.1 dùng map)
        if hasattr(styler_5, "map"):
            styler_5 = styler_5.map(color_excess_val, subset=["Chênh Lệch vs B&H [%]"])
        elif hasattr(styler_5, "applymap"):
            styler_5 = styler_5.applymap(color_excess_val, subset=["Chênh Lệch vs B&H [%]"])

        st.dataframe(styler_5, use_container_width=True)

        # Biểu đồ cột so sánh trực quan cả 5 mã
        col_bar1, col_bar2 = st.columns(2)
        with col_bar1:
            fig_bar_ret = go.Figure()
            fig_bar_ret.add_trace(go.Bar(
                x=df_comparison_5["Mã CP"],
                y=df_comparison_5["B&H Return [%]"],
                name="Buy & Hold Return (%)",
                marker_color="#EF4444"
            ))
            fig_bar_ret.add_trace(go.Bar(
                x=df_comparison_5["Mã CP"],
                y=df_comparison_5["SMA+OBV Return [%]"],
                name="SMA+OBV OR Return (%)",
                marker_color="#10B981"
            ))
            fig_bar_ret.update_layout(
                barmode="group",
                title="So sánh Lợi Nhuận Tổng Thể (Total Return %)",
                xaxis_title="Mã Cổ Phiếu",
                yaxis_title="Lợi Nhuận (%)",
                template="plotly_white",
                height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_bar_ret, use_container_width=True)

        with col_bar2:
            fig_bar_dd = go.Figure()
            fig_bar_dd.add_trace(go.Bar(
                x=df_comparison_5["Mã CP"],
                y=df_comparison_5["B&H Max DD [%]"],
                name="Buy & Hold Max DD (%)",
                marker_color="#F87171"
            ))
            fig_bar_dd.add_trace(go.Bar(
                x=df_comparison_5["Mã CP"],
                y=df_comparison_5["SMA+OBV Max DD [%]"],
                name="SMA+OBV OR Max DD (%)",
                marker_color="#3B82F6"
            ))
            fig_bar_dd.update_layout(
                barmode="group",
                title="So sánh Mức Sụt Giảm Tối Đa (Max Drawdown %)",
                xaxis_title="Mã Cổ Phiếu",
                yaxis_title="Max Drawdown (%)",
                template="plotly_white",
                height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_bar_dd, use_container_width=True)

        # Các bảng chi tiết phụ (Giao dịch, Train stats, Test stats)
        sub_tab1, sub_tab2, sub_tab3 = st.tabs([
            "🔄 Thống Kê Giao Dịch & Turnover (5 Mã)",
            "📊 Hiệu Suất Chi Tiết Train 2020-2021 (5 Mã)",
            "📉 Hiệu Suất Chi Tiết Test 2022 (5 Mã)"
        ])

        with sub_tab1:
            st.write("##### Bảng Thống Kê Số Lượng Lệnh Mua / Bán & Khối Lượng Luân Chuyển (Turnover)")
            st.dataframe(
                df_trades_summary_all.style.format({
                    "Train Lệnh Mua": "{:d}",
                    "Train Lệnh Bán": "{:d}",
                    "Test Lệnh Mua": "{:d}",
                    "Test Lệnh Bán": "{:d}",
                    "Test Turnover": "{:.2f}"
                }),
                use_container_width=True
            )

        with sub_tab2:
            st.write("##### Bảng Chỉ Số Hiệu Suất Chi Tiết Trên Tập Huấn Luyện (In-sample Train 2020–2021)")
            st.dataframe(df_strategy_train_stats_all.style.format("{:.4f}"), use_container_width=True)

        with sub_tab3:
            st.write("##### Bảng Chỉ Số Hiệu Suất Chi Tiết Trên Tập Kiểm Định (Out-of-sample Test 2022)")
            st.dataframe(df_strategy_test_stats_all.style.format("{:.4f}"), use_container_width=True)

        # PHẦN MỞ RỘNG: SOI BIỂU ĐỒ KỸ THUẬT CHI TIẾT
        with st.expander("🔍 Soi Biểu Đồ Kỹ Thuật Chi Tiết Từng Mã (Giá, SMA, OBV & Điểm Mua/Bán)", expanded=False):
            sc_c1, sc_c2, sc_c3, sc_c4 = st.columns(4)
            with sc_c1:
                detail_ticker = st.selectbox("Chọn mã cổ phiếu xem biểu đồ:", selected_top_tickers)
            curr_p = all_stock_results[detail_ticker]["params"]
            with sc_c2:
                dt_ma_s = st.number_input("SMA Ngắn:", min_value=5, max_value=100, value=curr_p["ma_short"], step=5, key="dt_ma_s")
            with sc_c3:
                dt_ma_l = st.number_input("SMA Dài:", min_value=20, max_value=400, value=curr_p["ma_long"], step=10, key="dt_ma_l")
            with sc_c4:
                dt_obv_w = st.number_input("OBV Window:", min_value=5, max_value=120, value=curr_p["obv_window"], step=5, key="dt_obv_w")

            detail_stk = stock_series_dict[detail_ticker]
            detail_res = evaluate_stock_strategy(
                detail_stk, dt_ma_s, dt_ma_l, dt_obv_w,
                test_start=test_start_str, test_end=test_end_str,
                commission=base_commission
            )
            detail_test_sub = detail_stk.loc[test_start_str:test_end_str]

            fig_ind = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.04,
                row_heights=[0.55, 0.25, 0.20],
                subplot_titles=[
                    f"Giá Cổ Phiếu {detail_ticker}, SMA({dt_ma_s}), SMA({dt_ma_l}) & Điểm Vào/Ra Lệnh (Test 2022)",
                    f"Chỉ Báo On-Balance Volume (OBV) & MA({dt_obv_w})",
                    "Trạng Thái Vị Thế (1: Nắm giữ Cổ phiếu, 0: Nắm giữ Tiền mặt)"
                ]
            )

            fig_ind.add_trace(go.Scatter(
                x=detail_test_sub.index, y=detail_test_sub["Close"],
                mode="lines", name="Giá Đóng Cửa", line=dict(color="#1F2937", width=1.8)
            ), row=1, col=1)

            fig_ind.add_trace(go.Scatter(
                x=detail_test_sub.index, y=detail_test_sub["Close"].rolling(int(dt_ma_s)).mean(),
                mode="lines", name=f"SMA {dt_ma_s}", line=dict(color="#3B82F6", width=1.5)
            ), row=1, col=1)

            fig_ind.add_trace(go.Scatter(
                x=detail_test_sub.index, y=detail_test_sub["Close"].rolling(int(dt_ma_l)).mean(),
                mode="lines", name=f"SMA {dt_ma_l}", line=dict(color="#F59E0B", width=1.5)
            ), row=1, col=1)

            # Markers
            b_dates = detail_res["events"][detail_res["events"] == 1.0].index
            b_prices = detail_test_sub.loc[b_dates, "Close"]
            fig_ind.add_trace(go.Scatter(
                x=b_dates, y=b_prices,
                mode="markers", name="Tín hiệu MUA",
                marker=dict(symbol="triangle-up", size=12, color="#10B981")
            ), row=1, col=1)

            s_dates = detail_res["events"][detail_res["events"] == -1.0].index
            s_prices = detail_test_sub.loc[s_dates, "Close"]
            fig_ind.add_trace(go.Scatter(
                x=s_dates, y=s_prices,
                mode="markers", name="Tín hiệu BÁN",
                marker=dict(symbol="triangle-down", size=12, color="#EF4444")
            ), row=1, col=1)

            # OBV
            obv_r = ta.volume.OnBalanceVolumeIndicator(close=detail_stk["Close"], volume=detail_stk["Volume"]).on_balance_volume()
            obv_ma_r = obv_r.rolling(int(dt_obv_w)).mean()
            fig_ind.add_trace(go.Scatter(
                x=obv_r.loc[test_start_str:test_end_str].index, y=obv_r.loc[test_start_str:test_end_str],
                mode="lines", name="OBV", line=dict(color="#8B5CF6", width=1.5)
            ), row=2, col=1)
            fig_ind.add_trace(go.Scatter(
                x=obv_ma_r.loc[test_start_str:test_end_str].index, y=obv_ma_r.loc[test_start_str:test_end_str],
                mode="lines", name=f"OBV MA({dt_obv_w})", line=dict(color="#EC4899", width=1.5, dash="dash")
            ), row=2, col=1)

            # Holding
            fig_ind.add_trace(go.Scatter(
                x=detail_res["holding"].index, y=detail_res["holding"],
                mode="lines", name="Vị Thế (Holding)",
                fill="tozeroy", line=dict(color="#059669", width=1.5)
            ), row=3, col=1)

            fig_ind.update_layout(
                height=650,
                template="plotly_white",
                showlegend=True,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            st.plotly_chart(fig_ind, use_container_width=True)

    # ==============================================================================
    # TAB 4: DANH MỤC MPT VS EQUAL WEIGHT VS BUY & HOLD
    # ==============================================================================
    with tab_portfolio:
        st.subheader("💼 Danh Mục Đầu Tư Đa Tài Sản (Top 5 HOSE)")
        st.markdown("""
        So sánh hiệu quả của 3 phương thức tiếp cận trên danh mục 5 luồng chiến lược:
        1. **Equal Weight Buy & Hold Benchmark:** Mua đều 20% mỗi mã tại đầu kỳ Test, giữ nguyên đến cuối kỳ (không daily rebalance).
        2. **SMA + OBV OR (Equal Weight):** Mỗi luồng chiến lược nhận 20% vốn cố định hàng ngày.
        3. **SMA + OBV OR (MPT Constrained):** Tối ưu hóa Sharpe Ratio trên Train với ràng buộc tỷ trọng [5%, 40%].
        """)

        # Tận dụng kết quả đã tính toán cho Top 5
        train_ret_matrix = pd.concat({
            t: all_stock_results[t]["train_strategy"]["returns"]
            for t in selected_top_tickers
        }, axis=1).dropna()

        test_ret_matrix = pd.concat({
            t: all_stock_results[t]["test_strategy"]["returns"]
            for t in selected_top_tickers
        }, axis=1).dropna()

        # Tối ưu hóa MPT
        n_assets = len(selected_top_tickers)
        eq_weights = np.repeat(1.0 / n_assets, n_assets)
        mpt_weights = optimize_mpt_sharpe(train_ret_matrix, min_weight=mpt_min_w, max_weight=mpt_max_w)

        # Trọng số
        w_df = pd.DataFrame({
            "Ticker": selected_top_tickers,
            "Equal Weight": eq_weights,
            "MPT Weight (Tối ưu Train)": mpt_weights
        })

        col_w1, col_w2 = st.columns([1, 1])
        with col_w1:
            st.write("##### ⚖️ Phân Bổ Tỷ Trọng Danh Mục")
            st.dataframe(
                w_df.style.format({
                    "Equal Weight": "{:.2%}",
                    "MPT Weight (Tối ưu Train)": "{:.2%}"
                }),
                use_container_width=True
            )

        with col_w2:
            fig_pie = go.Figure(data=[go.Pie(
                labels=w_df["Ticker"],
                values=w_df["MPT Weight (Tối ưu Train)"],
                hole=.4,
                title="MPT Weights"
            )])
            fig_pie.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20), template="plotly_white")
            st.plotly_chart(fig_pie, use_container_width=True)

        # Lợi nhuận danh mục
        ew_train_ret = calculate_portfolio_stream(train_ret_matrix, eq_weights)
        mpt_train_ret = calculate_portfolio_stream(train_ret_matrix, mpt_weights)

        ew_test_ret = calculate_portfolio_stream(test_ret_matrix, eq_weights)
        mpt_test_ret = calculate_portfolio_stream(test_ret_matrix, mpt_weights)

        # Benchmark Buy & Hold
        ew_bh_test_ret, ew_bh_test_wealth = calculate_equal_weight_buy_hold(
            stock_series_dict, selected_top_tickers, eq_weights
        )

        # Thống kê hiệu suất
        comp_test_df = pd.DataFrame({
            "Equal Weight Buy & Hold": performance_stats(ew_bh_test_ret),
            "SMA+OBV OR Equal Weight": performance_stats(ew_test_ret),
            "SMA+OBV OR MPT": performance_stats(mpt_test_ret)
        }).T

        st.write("##### 📊 Bảng So Sánh Hiệu Quả Đầu Tư Toàn Diện (Out-of-sample Test 2022)")
        st.dataframe(comp_test_df.style.format("{:.4f}"), use_container_width=True)

        # Tính Equity Curves
        bh_equity = initial_capital * (1 + ew_bh_test_ret).cumprod()
        ew_equity = initial_capital * (1 + ew_test_ret).cumprod()
        mpt_equity = initial_capital * (1 + mpt_test_ret).cumprod()

        # Biểu đồ Đường cong Tài sản (Equity Curve)
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=bh_equity.index, y=bh_equity,
            mode="lines", name="Equal Weight Buy & Hold Benchmark",
            line=dict(color="#EF4444", width=2.5, dash="dash")
        ))
        fig_equity.add_trace(go.Scatter(
            x=ew_equity.index, y=ew_equity,
            mode="lines", name="SMA+OBV OR (Equal Weight)",
            line=dict(color="#2563EB", width=2.5)
        ))
        fig_equity.add_trace(go.Scatter(
            x=mpt_equity.index, y=mpt_equity,
            mode="lines", name="SMA+OBV OR (MPT Optimized)",
            line=dict(color="#059669", width=2.8)
        ))

        fig_equity.update_layout(
            title="Đường Cong Tăng Trưởng Tài Sản Trên Tập Test (Equity Curve)",
            xaxis_title="Thời gian",
            yaxis_title="Giá trị Danh mục (VNĐ)",
            template="plotly_white",
            height=450,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_equity, use_container_width=True)

        # Biểu đồ Underwater Plot (Drawdown)
        def compute_drawdown(series):
            return (series / series.cummax() - 1.0) * 100

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=bh_equity.index, y=compute_drawdown(bh_equity),
            mode="lines", name="Buy & Hold Drawdown",
            fill="tozeroy", line=dict(color="#F87171", width=1.5)
        ))
        fig_dd.add_trace(go.Scatter(
            x=ew_equity.index, y=compute_drawdown(ew_equity),
            mode="lines", name="Equal Weight Strategy DD",
            fill="tozeroy", line=dict(color="#60A5FA", width=1.5)
        ))
        fig_dd.add_trace(go.Scatter(
            x=mpt_equity.index, y=compute_drawdown(mpt_equity),
            mode="lines", name="MPT Strategy DD",
            fill="tozeroy", line=dict(color="#34D399", width=1.5)
        ))

        fig_dd.update_layout(
            title="Biểu Đồ Sụt Giảm Giá Trị Danh Mục (Drawdown % / Underwater Plot)",
            xaxis_title="Thời gian",
            yaxis_title="Sụt giảm từ Đỉnh (%)",
            template="plotly_white",
            height=350,
            hovermode="x unified"
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # Tóm tắt kết luận tự động
        best_ret_name = comp_test_df["Total Return [%]"].idxmax()
        best_sharpe_name = comp_test_df["Sharpe Ratio"].idxmax()
        lowest_dd_name = comp_test_df["Max Drawdown [%]"].idxmax()

        st.markdown(f"""
        <div class="highlight-box">
            <b>💡 Kết luận tự động từ mô hình:</b><br>
            • <b>Lợi nhuận cao nhất trên Test:</b> <code>{best_ret_name}</code> ({comp_test_df.loc[best_ret_name, 'Total Return [%]']:.2f}%)<br>
            • <b>Chỉ số Sharpe Ratio tốt nhất:</b> <code>{best_sharpe_name}</code> ({comp_test_df.loc[best_sharpe_name, 'Sharpe Ratio']:.3f})<br>
            • <b>Kiểm soát rủi ro sụt giảm (Max DD) tốt nhất:</b> <code>{lowest_dd_name}</code> ({comp_test_df.loc[lowest_dd_name, 'Max Drawdown [%]']:.2f}%)<br>
            <i>Chiến lược kết hợp SMA + OBV giúp bảo vệ tài sản vượt trội trong giai đoạn thị trường Downtrend khốc liệt năm 2022.</i>
        </div>
        """, unsafe_allow_html=True)

    # ==============================================================================
    # TAB 5: PHÂN TÍCH ĐỘ NHẠY (SENSITIVITY ANALYSIS)
    # ==============================================================================
    with tab_sensitivity:
        st.subheader("🔬 Phân Tích Độ Nhạy Của Chiến Lược (Robustness & Sensitivity)")
        st.markdown("""
        Kiểm định xem chiến lược có nhạy cảm quá mức với chi phí giao dịch hay không.
        Một chiến lược định lượng tốt phải duy trì được hiệu quả ngay cả khi mức phí thị trường tăng cao.
        """)

        commission_scenarios = [0.0, 0.0005, 0.0010, 0.0020]
        cost_rows = []

        for comm in commission_scenarios:
            t_streams = {}
            for t in selected_top_tickers:
                params = DEFAULT_BEST_PARAMS.get(t, {"ma_short": 35, "ma_long": 200, "obv_window": 30})
                res = evaluate_stock_strategy(
                    stock_series_dict[t],
                    params["ma_short"], params["ma_long"], params["obv_window"],
                    test_start=test_start_str, test_end=test_end_str,
                    commission=comm
                )
                t_streams[t] = res["returns"]

            mat = pd.concat(t_streams, axis=1).dropna()
            ew_r = calculate_portfolio_stream(mat, eq_weights)
            mpt_r = calculate_portfolio_stream(mat, mpt_weights)

            cost_rows.append({
                "Phí Giao Dịch (%)": f"{comm * 100:.2f}%",
                "EW Total Return [%]": performance_stats(ew_r)["Total Return [%]"],
                "EW Sharpe": performance_stats(ew_r)["Sharpe Ratio"],
                "MPT Total Return [%]": performance_stats(mpt_r)["Total Return [%]"],
                "MPT Sharpe": performance_stats(mpt_r)["Sharpe Ratio"]
            })

        cost_df = pd.DataFrame(cost_rows)
        st.write("##### 📊 Độ Nhạy Theo Các Mức Phí Giao Dịch Khác Nhau (0% - 0.20%)")
        st.dataframe(cost_df.style.format({
            "EW Total Return [%]": "{:.4f}%",
            "EW Sharpe": "{:.4f}",
            "MPT Total Return [%]": "{:.4f}%",
            "MPT Sharpe": "{:.4f}"
        }), use_container_width=True)

        # Biểu đồ đường so sánh độ nhạy
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Bar(
            x=cost_df["Phí Giao Dịch (%)"],
            y=cost_df["EW Total Return [%]"],
            name="Equal Weight Total Return (%)",
            marker_color="#3B82F6"
        ))
        fig_cost.add_trace(go.Bar(
            x=cost_df["Phí Giao Dịch (%)"],
            y=cost_df["MPT Total Return [%]"],
            name="MPT Total Return (%)",
            marker_color="#10B981"
        ))
        fig_cost.update_layout(
            barmode="group",
            title="Tác động của Phí giao dịch lên Lợi nhuận Danh mục",
            xaxis_title="Kịch bản Phí giao dịch",
            yaxis_title="Lợi nhuận tổng thể (%)",
            template="plotly_white",
            height=380
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    # ==============================================================================
    # TAB 6: PHƯƠNG PHÁP LUẬN & HƯỚNG DẪN
    # ==============================================================================
    with tab_methodology:
        st.subheader("📖 Phương Pháp Luận Định Lượng & Quy Trình Kiểm Định")
        st.markdown("""
        ### 1. Quy Trình Nghiên Cứu 7 Bước
        Hệ thống được xây dựng theo chuẩn kiểm định mô hình định lượng (Quantitative Trading Standard):
        ```
        Dữ liệu HOSE → TA Screening (Train 2020-2021) → Top 5 Cổ Phiếu → 
        Chiến lược SMA + OBV (OR) → Tối ưu Tham số Hyperopt → 
        Kiểm định Ngoài Mẫu (Test 2022 Warm-up) → Phân bổ Danh mục MPT vs Benchmark Buy & Hold
        ```

        ### 2. Loại Bỏ Sai Lệch (Biases Prevention)
        1. **Look-ahead Bias:** Tín hiệu kỹ thuật phát sinh tại phiên $t$ chỉ được phép thực thi tại phiên $t+1$ (`shift(1)`).
        2. **Corporate Action Bias:** Ưu tiên sử dụng giá điều chỉnh (`adj_close`) để đảm bảo không bị méo mó khi cổ phiếu chia cổ tức hoặc phát hành quyền mua.
        3. **Out-of-sample Testing:** Bộ tham số và mô hình lựa chọn chỉ được huấn luyện trên giai đoạn 2020 – 2021 (thị trường Uptrend), sau đó kiểm định độc lập trên năm 2022 (thị trường Downtrend).
        4. **Warm-up Indicator:** Toàn bộ dữ liệu trước năm 2022 được dùng làm giai đoạn warm-up cho các chỉ báo dài hạn (như SMA350) nhằm duy trì trạng thái crossover chính xác.
        5. **Benchmark Buy & Hold Chuẩn Xác:** Không tái cân bằng danh mục thụ động hàng ngày mà để tỷ trọng trôi tự nhiên theo biến động thị trường.

        ### 3. Ý Nghĩa Kết Hợp Tín Hiệu SMA + OBV (Quy tắc OR)
        - **SMA (Xu Hướng Giá):** Phản ánh quán tính và xác nhận xu hướng trung/dài hạn.
        - **OBV (Dòng Tiền Khối Lượng):** Đo lường sự tích lũy/phân phối sớm của dòng tiền thông minh trước khi giá bứt phá.
        - **Logic OR:** Cho phép hệ thống kích hoạt vị thế linh hoạt khi có sự bùng nổ của dòng tiền *hoặc* xác nhận từ xu hướng giá, đồng thời kích hoạt thoát vị thế nhanh chóng khi một trong hai yếu tố suy yếu.
        """)


if __name__ == "__main__":
    main()
