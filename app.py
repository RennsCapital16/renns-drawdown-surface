
import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re

# ==========================================
# RENNS CAPITAL
# QUANTITATIVE RESEARCH TERMINAL
# DRAWDOWN SURFACES V5.0
# ==========================================

st.set_page_config(
    page_title="Renns Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# USER INTERFACE
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #080B10;
    color: #FFFFFF;
}
[data-testid="stSidebar"] {
    background-color: #10141B;
}
[data-testid="stMetric"] {
    background-color: #111720;
    padding: 18px;
    border: 1px solid #28303D;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

st.title("RENNS CAPITAL")
st.caption("QUANTITATIVE RESEARCH TERMINAL")
st.divider()

st.sidebar.title("RENNS TERMINAL")
st.sidebar.caption("QUANTITATIVE ANALYTICS")

st.sidebar.divider()

ticker = st.sidebar.text_input(
    "Asset Ticker",
    value="BBCA.JK"
).strip().upper()

paths = st.sidebar.selectbox(
    "Monte Carlo Paths",
    [1000, 5000, 10000, 25000, 50000],
    index=2
)

days = st.sidebar.selectbox(
    "Trading Horizon",
    [21, 63, 126, 252, 504],
    index=3
)

years = st.sidebar.selectbox(
    "Historical Data (Years)",
    [1, 2, 3, 5],
    index=0
)

st.sidebar.divider()

st.sidebar.caption("DATA PROVIDER: EODHD")
st.sidebar.caption("MODEL: GEOMETRIC BROWNIAN MOTION")
st.sidebar.caption("ENGINE VERSION: 5.0")

run = st.sidebar.button(
    "RUN SIMULATION",
    type="primary",
    use_container_width=True
)

# ==========================================
# EODHD MARKET DATA ENGINE
# ==========================================

@st.cache_data(ttl=3600, max_entries=32)
def get_market_data(ticker, years):

    api_key = st.secrets["EODHD_API_KEY"]

    start = (
        pd.Timestamp.today()
        - pd.DateOffset(years=years)
    ).strftime("%Y-%m-%d")

    response = requests.get(
        f"https://eodhd.com/api/eod/{ticker}",
        params={
            "api_token": api_key,
            "fmt": "json",
            "period": "d",
            "from": start
        },
        timeout=20
    )

    if response.status_code != 200:
        raise ValueError(
            f"EODHD HTTP Error: {response.status_code}"
        )

    data = response.json()

    if not isinstance(data, list) or not data:
        raise ValueError("Market data unavailable")

    df = pd.DataFrame(data)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(subset=["date"])

    df = df.set_index("date").sort_index()

    df = df[
        ~df.index.duplicated(keep="last")
    ]

    close = pd.to_numeric(
        df["adjusted_close"],
        errors="coerce"
    )

    close = close[
        np.isfinite(close) & (close > 0)
    ]

    return close


# ==========================================
# MONTE CARLO ENGINE
# ==========================================

def run_monte_carlo(close, paths, days):

    returns = np.log(close).diff().dropna()

    volatility = returns.std() * np.sqrt(252)

    mu = (
        returns.mean() * 252
        + 0.5 * volatility**2
    )

    if not np.isfinite(volatility) or not np.isfinite(mu):
        raise ValueError("Invalid market parameters")

    rng = np.random.default_rng(42)

    thresholds = np.arange(1, 81) / 100
    horizons = np.arange(1, days + 1)

    log_price = np.zeros(paths)
    peak = np.zeros(paths)
    mdd = np.zeros(paths)

    surface = np.zeros(
        (len(thresholds), days)
    )

    drift = (
        mu - 0.5 * volatility**2
    ) / 252

    diffusion = volatility / np.sqrt(252)

    for day in range(days):

        log_price += (
            drift + diffusion
            * rng.standard_normal(paths)
        )

        peak = np.maximum(
            peak, log_price
        )

        drawdown = -np.expm1(
            log_price - peak
        )

        mdd = np.maximum(
            mdd, drawdown
        )

        ordered = np.sort(mdd)

        surface[:, day] = (
            paths - np.searchsorted(
                ordered,
                thresholds,
                side="left"
            )
        ) / paths * 100

    # Mathematical validation

    assert np.all(np.isfinite(surface))
    assert np.all(surface >= 0)
    assert np.all(surface <= 100)

    assert np.all(
        np.diff(surface, axis=1) >= -1e-9
    )

    assert np.all(
        np.diff(surface, axis=0) <= 1e-9
    )

    return (
        volatility,
        mu,
        thresholds,
        horizons,
        surface,
        mdd
    )


# ==========================================
# EXECUTION
# ==========================================

st.header("Drawdown Surfaces")

st.caption(
    "Monte Carlo probability analysis of "
    "peak-to-trough financial drawdowns."
)

if run:

    if not re.fullmatch(
        r"[A-Z0-9._-]{1,30}", ticker
    ):
        st.error("Invalid asset ticker.")
        st.stop()

    with st.spinner(
        "Running Renns Quantitative Engine..."
    ):

        try:

            close = get_market_data(
                ticker, years
            )

            if len(close) < 60:
                st.error(
                    "Insufficient historical data."
                )
                st.stop()

            (
                volatility,
                mu,
                thresholds,
                horizons,
                surface,
                mdd
            ) = run_monte_carlo(
                close, paths, days
            )

            st.session_state["result"] = {
                "ticker": ticker,
                "paths": paths,
                "days": days,
                "close": close,
                "volatility": volatility,
                "mu": mu,
                "thresholds": thresholds,
                "horizons": horizons,
                "surface": surface,
                "mdd": mdd
            }

        except Exception:

            st.error(
                "Simulation failed. Check EODHD "
                "API credentials, ticker, "
                "subscription and market data."
            )

            st.stop()


# ==========================================
# DISPLAY RESULTS
# ==========================================

if "result" in st.session_state:

    result = st.session_state["result"]

    asset = result["ticker"]
    close = result["close"]
    mdd = result["mdd"]
    surface = result["surface"]
    thresholds = result["thresholds"]
    horizons = result["horizons"]
    volatility = result["volatility"]
    mu = result["mu"]

    expected_mdd = np.mean(mdd)
    median_mdd = np.median(mdd)

    var95 = np.quantile(mdd, 0.95)

    tail95 = np.mean(
        np.sort(mdd)[
            -max(1, int(len(mdd) * 0.05)):
        ]
    )

    historical_dd = 1 - close / close.cummax()

    # ASSET INFORMATION

    st.subheader(asset)

    st.caption(
        f"EODHD | "
        f"{len(close)} Historical Observations | "
        f"Last Data: {close.index[-1].date()} | "
        f"{result['paths']:,} Simulations | "
        f"{result['days']} Trading Days"
    )

    # RISK METRICS

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Annual Volatility",
        f"{volatility:.2%}"
    )

    c2.metric(
        "Expected Max Drawdown",
        f"{expected_mdd:.2%}"
    )

    c3.metric(
        "Drawdown VaR 95%",
        f"{var95:.2%}"
    )

    c4.metric(
        "Tail Risk 95%",
        f"{tail95:.2%}"
    )

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "Estimated Annual Drift",
        f"{mu:.2%}"
    )

    c6.metric(
        "Median Max Drawdown",
        f"{median_mdd:.2%}"
    )

    c7.metric(
        "Historical Max Drawdown",
        f"{historical_dd.max():.2%}"
    )

    c8.metric(
        "Probability DD >= 20%",
        f"{np.mean(mdd >= 0.20):.2%}"
    )

    st.divider()

    # ======================================
    # INTERACTIVE ANALYTICS TABS
    # ======================================

    tab1, tab2, tab3, tab4 = st.tabs([
        "3D Drawdown Surface",
        "Risk Heatmap",
        "Historical Drawdown",
        "Risk Report"
    ])

    # 3D SURFACE

    with tab1:

        fig = go.Figure(
            data=[go.Surface(
                x=horizons,
                y=thresholds * 100,
                z=surface,
                colorscale="Turbo",
                cmin=0,
                cmax=100,
                colorbar=dict(
                    title="Probability %"
                ),
                hovertemplate=(
                    "Trading Days: %{x}<br>"
                    "Drawdown: %{y:.1f}%<br>"
                    "Probability: %{z:.2f}%"
                    "<extra></extra>"
                )
            )]
        )

        fig.update_layout(
            title=f"{asset} | Drawdown Surfaces",
            template="plotly_dark",
            paper_bgcolor="#080B10",
            font=dict(
                family="Menlo, monospace",
                color="white"
            ),
            scene=dict(
                xaxis_title="Trading Days",
                yaxis_title="Drawdown (%)",
                zaxis_title="Probability (%)",
                zaxis=dict(range=[0, 100]),
                camera=dict(
                    eye=dict(x=1.7, y=1.7, z=1.0)
                )
            ),
            height=700,
            margin=dict(l=0,r=0,b=0,t=50)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # RISK HEATMAP

    with tab2:

        heatmap = go.Figure(
            go.Heatmap(
                x=horizons,
                y=thresholds * 100,
                z=surface,
                colorscale="Turbo",
                zmin=0,
                zmax=100,
                colorbar=dict(
                    title="Probability %"
                )
            )
        )

        heatmap.update_layout(
            title=f"{asset} | Risk Heatmap",
            template="plotly_dark",
            paper_bgcolor="#080B10",
            xaxis_title="Trading Days",
            yaxis_title="Drawdown (%)",
            height=550
        )

        st.plotly_chart(
            heatmap,
            use_container_width=True
        )

    # HISTORICAL DRAWDOWN

    with tab3:

        historical_fig = go.Figure(
            go.Scatter(
                x=historical_dd.index,
                y=-historical_dd * 100,
                fill="tozeroy",
                line=dict(color="#EF4444")
            )
        )

        historical_fig.update_layout(
            title=f"{asset} | Historical Drawdown",
            template="plotly_dark",
            paper_bgcolor="#080B10",
            yaxis_title="Drawdown (%)",
            height=500
        )

        st.plotly_chart(
            historical_fig,
            use_container_width=True
        )

    # QUANTITATIVE RISK REPORT

    with tab4:

        risk_table = pd.DataFrame({
            "Drawdown Threshold": [
                f"{t:.0%}"
                for t in [.05,.1,.2,.3,.4,.5,.6,.7,.8]
            ],
            "Probability": [
                f"{np.mean(mdd >= t):.2%}"
                for t in [.05,.1,.2,.3,.4,.5,.6,.7,.8]
            ]
        })

        st.subheader("Drawdown Exceedance Probability")

        st.dataframe(
            risk_table,
            use_container_width=True,
            hide_index=True
        )

        export = pd.DataFrame(
            surface,
            index=thresholds * 100,
            columns=horizons
        )

        export.index.name = "Drawdown (%)"

        st.download_button(
            label="Download Quantitative Risk Data",
            data=export.to_csv(),
            file_name=f"{asset}_Drawdown_Risk.csv",
            mime="text/csv"
        )

    st.success(
        "RENNS QUANT ENGINE: MATHEMATICAL CHECKS PASSED"
    )

else:

    st.info(
        "Enter an asset ticker in the sidebar "
        "and click RUN SIMULATION."
    )

st.divider()

st.caption(
    "RENNS CAPITAL | QUANTITATIVE RESEARCH"
)

st.caption(
    "Simulations are model-based estimates, "
    "not guarantees of future market performance."
)
