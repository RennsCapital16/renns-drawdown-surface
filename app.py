
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

        # MONTE CARLO PRICE PATHS

        st.subheader("Monte Carlo Price Paths")

        display_paths = st.selectbox(
            "Displayed Simulation Paths",
            [25, 50, 100, 200, 500],
            index=2
        )

        total_paths = result["paths"]
        trading_days = result["days"]

        n = min(display_paths, total_paths)

        rng_paths = np.random.default_rng(42)

        dt = 1 / 252

        drift = (
            mu - 0.5 * volatility**2
        ) * dt

        diffusion = volatility * np.sqrt(dt)

        log_values = np.zeros(n)

        price_paths = np.zeros(
            (n, trading_days + 1)
        )

        price_paths[:, 0] = 100

        for day in range(1, trading_days + 1):

            shocks = rng_paths.standard_normal(
                total_paths
            )[:n]

            log_values += (
                drift + diffusion * shocks
            )

            price_paths[:, day] = (
                100 * np.exp(log_values)
            )

        path_fig = go.Figure()

        for i, path in enumerate(price_paths):

            color = (
                "rgba(16,185,129,0.35)"
                if path[-1] >= 100
                else "rgba(239,68,68,0.35)"
            )

            path_fig.add_trace(
                go.Scattergl(
                    x=np.arange(trading_days + 1),
                    y=path,
                    mode="lines",
                    line=dict(
                        color=color,
                        width=0.8
                    ),
                    name=f"Simulation {i+1}",
                    showlegend=False,
                    hovertemplate=(
                        f"Path: {i+1}<br>"
                        "Day: %{x}<br>"
                        "Price Index: %{y:.2f}"
                        "<extra></extra>"
                    )
                )
            )

        path_fig.add_hline(
            y=100,
            line_dash="dash",
            line_color="white",
            annotation_text="Initial Price"
        )

        path_fig.update_layout(
            title=f"{asset} | Monte Carlo Price Paths",
            template="plotly_dark",
            paper_bgcolor="#080B10",
            plot_bgcolor="#080B10",
            xaxis_title="Trading Days",
            yaxis_title="Normalized Price Index",
            font=dict(
                family="Menlo, monospace",
                color="white"
            ),
            height=550,
            hovermode="closest",
            margin=dict(
                l=20, r=20, t=60, b=40
            )
        )

        st.plotly_chart(
            path_fig,
            use_container_width=True
        )

        st.caption(
            f"Displaying {n:,} of "
            f"{total_paths:,} simulated paths. "
            "Initial price normalized to 100."
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

    # ======================================
    # ADVANCED MONTE CARLO ANALYTICS V6.0
    # ======================================

    st.divider()

    st.header("Advanced Monte Carlo Analytics")

    st.caption(
        "Forward-looking return distribution "
        "and probabilistic price projections."
    )

    @st.cache_data(max_entries=8, ttl=3600)
    def calculate_projection(mu, vol, paths, days):

        rng = np.random.default_rng(42)

        drift = (mu - 0.5 * vol**2) / 252
        diffusion = vol / np.sqrt(252)

        log_prices = np.zeros(paths)

        quantiles = np.array([
            0.05, 0.25, 0.50, 0.75, 0.95
        ])

        bands = np.ones((5, days + 1)) * 100

        for day in range(1, days + 1):

            log_prices += (
                drift
                + diffusion
                * rng.standard_normal(paths)
            )

            bands[:, day] = 100 * np.exp(
                np.quantile(
                    log_prices,
                    quantiles
                )
            )

        terminal_returns = np.expm1(
            log_prices
        )

        return bands, terminal_returns

    projection, final_returns = calculate_projection(
        float(mu),
        float(volatility),
        result["paths"],
        result["days"]
    )

    p5, p25, p50, p75, p95 = projection

    projection_days = np.arange(
        result["days"] + 1
    )

    probability_profit = np.mean(
        final_returns > 0
    )

    probability_loss = np.mean(
        final_returns < 0
    )

    probability_flat = np.mean(
        final_returns == 0
    )

    # ADVANCED METRICS

    a1, a2, a3, a4 = st.columns(4)

    a1.metric(
        "Probability of Profit",
        f"{probability_profit:.2%}"
    )

    a2.metric(
        "Probability of Loss",
        f"{probability_loss:.2%}"
    )

    a3.metric(
        "Mean Simulated Return",
        f"{final_returns.mean():.2%}"
    )

    a4.metric(
        "Median Simulated Return",
        f"{np.median(final_returns):.2%}"
    )

    st.caption(
        f"Probability of unchanged value: "
        f"{probability_flat:.2%}. "
        "Excludes transaction costs and taxes."
    )

    advanced_tab1, advanced_tab2, advanced_tab3 = st.tabs([
        "Confidence Bands",
        "Return Distribution",
        "Return Percentiles"
    ])

    # ======================================
    # CONFIDENCE BANDS
    # ======================================

    with advanced_tab1:

        band_fig = go.Figure()

        band_fig.add_trace(
            go.Scatter(
                x=projection_days,
                y=p5,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            )
        )

        band_fig.add_trace(
            go.Scatter(
                x=projection_days,
                y=p95,
                fill="tonexty",
                fillcolor="rgba(59,130,246,0.15)",
                mode="lines",
                line=dict(width=0),
                name="90% Prediction Interval"
            )
        )

        band_fig.add_trace(
            go.Scatter(
                x=projection_days,
                y=p25,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            )
        )

        band_fig.add_trace(
            go.Scatter(
                x=projection_days,
                y=p75,
                fill="tonexty",
                fillcolor="rgba(59,130,246,0.30)",
                mode="lines",
                line=dict(width=0),
                name="50% Prediction Interval"
            )
        )

        band_fig.add_trace(
            go.Scatter(
                x=projection_days,
                y=p50,
                mode="lines",
                line=dict(
                    color="#60A5FA",
                    width=2.5
                ),
                name="Median Projection"
            )
        )

        band_fig.add_hline(
            y=100,
            line_dash="dash",
            line_color="#AAAAAA",
            annotation_text="Initial Value"
        )

        band_fig.update_layout(
            title=f"{asset} | Monte Carlo Price Projection",
            template="plotly_dark",
            paper_bgcolor="#080B10",
            plot_bgcolor="#080B10",
            xaxis_title="Trading Days",
            yaxis_title="Normalized Value Index",
            height=550,
            hovermode="x unified",
            font=dict(
                family="Menlo, monospace"
            )
        )

        st.plotly_chart(
            band_fig,
            use_container_width=True,
            key="renns_projection_bands"
        )

        st.caption(
            "Pointwise simulated prediction intervals, "
            "not confidence intervals for the mean. "
            "Initial value normalized to 100."
        )

    # ======================================
    # TERMINAL RETURN DISTRIBUTION
    # ======================================

    with advanced_tab2:

        dist_fig = go.Figure()

        returns_pct = final_returns * 100

        edges = np.histogram_bin_edges(
            returns_pct,
            bins=80
        )

        negative_counts, _ = np.histogram(
            returns_pct[returns_pct < 0],
            bins=edges
        )

        positive_counts, _ = np.histogram(
            returns_pct[returns_pct >= 0],
            bins=edges
        )

        centers = (
            edges[:-1] + edges[1:]
        ) / 2

        widths = np.diff(edges)

        dist_fig.add_trace(
            go.Bar(
                x=centers,
                y=negative_counts,
                width=widths,
                marker_color="#EF4444",
                name="Negative Returns"
            )
        )

        dist_fig.add_trace(
            go.Bar(
                x=centers,
                y=positive_counts,
                width=widths,
                marker_color="#10B981",
                name="Nonnegative Returns"
            )
        )

        dist_fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="white"
        )

        dist_fig.update_layout(
            title=f"{asset} | Terminal Return Distribution",
            template="plotly_dark",
            paper_bgcolor="#080B10",
            plot_bgcolor="#080B10",
            xaxis_title="Simulated Return (%)",
            yaxis_title="Number of Scenarios",
            barmode="stack",
            bargap=0,
            height=500
        )

        st.plotly_chart(
            dist_fig,
            use_container_width=True,
            key="renns_return_distribution"
        )

    # ======================================
    # RETURN PERCENTILES
    # ======================================

    with advanced_tab3:

        percentile_values = np.quantile(
            final_returns,
            [0.05, 0.25, 0.50, 0.75, 0.95]
        )

        scenario_report = pd.DataFrame({

            "Scenario": [
                "Pessimistic",
                "Lower Quartile",
                "Median",
                "Upper Quartile",
                "Optimistic"
            ],

            "Percentile": [
                "P5", "P25", "P50", "P75", "P95"
            ],

            "Simulated Return": [
                f"{v:.2%}"
                for v in percentile_values
            ],

            "Final Value Index": [
                round(100 * (1 + v), 2)
                for v in percentile_values
            ]

        })

        st.dataframe(
            scenario_report,
            hide_index=True,
            use_container_width=True
        )

        st.download_button(
            "Download Return Scenarios",
            scenario_report.to_csv(index=False),
            file_name=f"{asset}_Return_Scenarios.csv",
            mime="text/csv",
            key="renns_return_export"
        )

    st.caption(
        "RENNS QUANT V6.0 | "
        "Model-based probabilistic estimates. "
        "Not guaranteed future returns."
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
