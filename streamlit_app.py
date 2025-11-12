# =======================================================
#  --- FUNCIONES DE YFINANCE ---
# =======================================================
import yfinance as yf

def yf_get_expirations(ticker: str):
    """Obtiene expiraciones disponibles desde yFinance"""
    try:
        return yf.Ticker(ticker).options
    except Exception:
        return []

def yf_get_calls_puts_for_exp(ticker: str, exp: str):
    """Obtiene las cadenas de opciones (calls y puts)"""
    try:
        oc = yf.Ticker(ticker).option_chain(exp)
        return oc.calls, oc.puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

# =======================================================
#  --- FUNCION PARA PRECIO SPOT ---
# =======================================================
def get_spot_price(ticker: str):
    """Obtiene el precio spot actual del ticker"""
    try:
        data = yf.Ticker(ticker).history(period="1d", interval="1m")
        if not data.empty:
            return float(data["Close"].iloc[-1])
        else:
            return float("nan")
    except Exception:
        return float("nan")
# =======================================================
#  --- CONSTANTES Y FUNCION PARA CALCULAR EXPOSICIONES ---
# =======================================================
import numpy as np
from scipy.stats import norm

DEFAULT_R = 0.05  # tasa libre de riesgo
DEFAULT_Q = 0.00  # dividend yield

def compute_exposures(calls, puts, spot, dte, r, q):
    """Calcula exposición gamma y delta de calls y puts"""
    if calls.empty or puts.empty:
        return pd.DataFrame()

    sigma_calls = calls["impliedVolatility"].fillna(0.3)
    sigma_puts = puts["impliedVolatility"].fillna(0.3)

    # --- Black-Scholes greeks ---
    def bs_d1(S, K, T, r, q, sigma):
        return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    def bs_gamma(S, K, T, r, q, sigma):
        d1 = bs_d1(S, K, T, r, q, sigma)
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def bs_delta_call(S, K, T, r, q, sigma):
        d1 = bs_d1(S, K, T, r, q, sigma)
        return np.exp(-q * T) * norm.cdf(d1)

    def bs_delta_put(S, K, T, r, q, sigma):
        d1 = bs_d1(S, K, T, r, q, sigma)
        return -np.exp(-q * T) * norm.cdf(-d1)

    T = dte / 365

    calls["gamma"] = bs_gamma(spot, calls["strike"], T, r, q, sigma_calls)
    puts["gamma"] = bs_gamma(spot, puts["strike"], T, r, q, sigma_puts)

    calls["delta"] = bs_delta_call(spot, calls["strike"], T, r, q, sigma_calls)
    puts["delta"] = bs_delta_put(spot, puts["strike"], T, r, q, sigma_puts)

    calls["call_gex"] = calls["gamma"] * spot * spot * 100 * calls["openInterest"]
    puts["put_gex"] = -puts["gamma"] * spot * spot * 100 * puts["openInterest"]

    calls["call_dex"] = calls["delta"] * spot * 100 * calls["openInterest"]
    puts["put_dex"] = puts["delta"] * spot * 100 * puts["openInterest"]

    df = pd.DataFrame({
        "strike": calls["strike"],
        "call_gex": calls["call_gex"],
        "put_gex": puts["put_gex"],
        "call_dex": calls["call_dex"],
        "put_dex": puts["put_dex"]
    })

    df["net_gex"] = df["call_gex"] + df["put_gex"]
    df["net_dex"] = df["call_dex"] + df["put_dex"]

    return df

def plot_gex_dex(df, spot):
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="Sin datos para graficar GEX/DEX")
        return fig

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        subplot_titles=("Gamma Exposure (USD)", "Delta Exposure (USD)")
    )

    # ---- Colores base ----
    color_calls = "lightgray"
    color_puts = "#1f77b4"   # azul intenso
    color_net = "red"
    color_spot = "yellow"

    # ---- Gamma Exposure (USD) ----
    fig.add_bar(
        x=df["strike"],
        y=df["call_gex"],
        name="Calls GEX",
        marker_color=color_calls,
        row=1, col=1
    )
    fig.add_bar(
        x=df["strike"],
        y=df["put_gex"],  # Puts abajo
        name="Puts GEX",
        marker_color=color_puts,
        row=1, col=1
    )
    fig.add_scatter(
        x=df["strike"],
        y=df["net_gex"],
        mode="lines+markers",
        name="Net GEX",
        line=dict(color=color_net, width=2),
        marker=dict(size=4),
        row=1, col=1
    )

    # ---- Delta Exposure (USD) ----
    fig.add_bar(
        x=df["strike"],
        y=df["call_dex"],
        name="Calls DEX",
        marker_color=color_calls,
        row=2, col=1
    )
    fig.add_bar(
        x=df["strike"],
        y=df["put_dex"],
        name="Puts DEX",
        marker_color=color_puts,
        row=2, col=1
    )
    fig.add_scatter(
        x=df["strike"],
        y=df["net_dex"],
        mode="lines+markers",
        name="Net DEX",
        line=dict(color=color_net, width=2),
        marker=dict(size=4),
        row=2, col=1
    )

    # ---- Spot Line ----
    fig.add_vline(
        x=spot,
        line_color=color_spot,
        line_dash="dash",
        annotation_text=f"Spot {spot:.2f}",
        annotation_position="top right"
    )

    # ---- Gamma extrema ----
    max_idx = df["net_gex"].idxmax()
    min_idx = df["net_gex"].idxmin()
    max_strike, min_strike = df.loc[max_idx, "strike"], df.loc[min_idx, "strike"]
    max_val, min_val = df.loc[max_idx, "net_gex"], df.loc[min_idx, "net_gex"]

    # Anotaciones de máximo y mínimo GEX
    fig.add_annotation(
        x=max_strike, y=max_val,
        text=f"▲ {max_strike:.1f} ({max_val/1e6:.1f}M)",
        showarrow=True, arrowhead=2, ax=20, ay=-40,
        font=dict(color="white", size=11),
        bgcolor="rgba(0,0,0,0.5)"
    )
    fig.add_annotation(
        x=min_strike, y=min_val,
        text=f"▼ {min_strike:.1f} ({min_val/1e6:.1f}M)",
        showarrow=True, arrowhead=2, ax=-20, ay=40,
        font=dict(color="white", size=11),
        bgcolor="rgba(0,0,0,0.5)"
    )

    # ---- Layout ----
    fig.update_layout(
        template="plotly_dark",
        barmode="overlay",  # para que puts vayan abajo
        height=750,
        bargap=0.1,
        title="GEX/DEX ibit (agregado)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(size=12),
        plot_bgcolor="#0e1621",
        paper_bgcolor="#0e1621",
    )

    # Líneas horizontales
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="white")
    return fig


# =======================================================
#  --- STREAMLIT UI ---
# =======================================================
import streamlit as st
import pandas as pd

# Asegúrate de importar tus funciones y dependencias necesarias
# Por ejemplo:
# from tu_modulo import yf_get_expirations, yf_get_calls_puts_for_exp, get_spot_price, compute_exposures, DEFAULT_R, DEFAULT_Q

st.set_page_config(page_title="Gamma Quant Viewer", layout="wide")
st.title("📊 Gamma Quant Viewer")

ticker = st.text_input("Ticker (ej: TSLA, AAPL, SPY):", value="TSLA")

# Obtener expiraciones desde yFinance
expirations = yf_get_expirations(ticker)

if expirations:
    exp_dates = pd.to_datetime(expirations)
    today = pd.Timestamp.today()
    exp_dte = pd.Series((exp_dates - today)).dt.days
    exp_df = pd.DataFrame({"exp": expirations, "dte": exp_dte})

    exp_type = st.selectbox(
        "Selecciona tipo de expiración:",
        ["Weekly", "2 Semanas", "Mensual", "Todas"]
    )

    if exp_type == "Weekly":
        filtered = exp_df[exp_df["dte"] <= 7]
    elif exp_type == "2 Semanas":
        filtered = exp_df[(exp_df["dte"] > 7) & (exp_df["dte"] <= 14)]
    elif exp_type == "Mensual":
        filtered = exp_df[(exp_df["dte"] > 14) & (exp_df["dte"] <= 35)]
    else:
        filtered = exp_df

    if not filtered.empty:
        selected_exp = st.selectbox(
            "Selecciona expiración específica:",
            options=filtered["exp"].tolist()
        )
    else:
        selected_exp = st.selectbox(
            "Selecciona expiración específica:",
            options=exp_df["exp"].tolist()
        )
else:
    st.warning("No se pudieron obtener expiraciones para este ticker.")
    selected_exp = None

if selected_exp and st.button("Calcular GEX/DEX"):
    with st.spinner("Obteniendo datos y calculando exposiciones..."):
        calls, puts = yf_get_calls_puts_for_exp(ticker, selected_exp)
        if calls is None or puts is None or calls.empty or puts.empty:
            st.error("No se pudieron obtener datos de opciones.")
        else:
            spot = get_spot_price(ticker)
            df = compute_exposures(calls, puts, spot, 7, DEFAULT_R, DEFAULT_Q)


            st.plotly_chart(plot_gex_dex(df, spot), width="stretch")

            # =======================================================
            #  --- MAPA DE OPEN INTEREST (Barras Calls/Puts) ---
            # =======================================================
            st.subheader("📊 Mapa de Open Interest")

            import plotly.graph_objects as go

            oi_fig = go.Figure()

            # Barras de Calls (positivas)
            oi_fig.add_bar(
                x=calls["strike"],
                y=calls["openInterest"],
                name="Calls OI",
                marker_color="lightgray"
            )

            # Barras de Puts (negativas, invertidas para contraste visual)
            oi_fig.add_bar(
                x=puts["strike"],
                y=-puts["openInterest"],
                name="Puts OI",
                marker_color="#1f77b4"
            )

            # Línea vertical en el precio spot
            oi_fig.add_vline(
                x=spot,
                line_color="yellow",
                line_dash="dash",
                annotation_text=f"Spot {spot:.2f}",
                annotation_position="top right"
            )

            # Configuración visual
            oi_fig.update_layout(
                template="plotly_dark",
                barmode="overlay",
                height=500,
                bargap=0.1,
                title="Open Interest por Strike",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                font=dict(size=12),
                plot_bgcolor="#0e1621",
                paper_bgcolor="#0e1621",
                yaxis_title="Open Interest (positivo Calls / negativo Puts)",
                xaxis_title="Strike"
            )


            st.plotly_chart(oi_fig, width="stretch")

            # =======================================================
            #  --- MAPA COMBINADO OI vs NET GEX ---
            # =======================================================
            st.subheader("🧭 Mapa combinado OI vs Net GEX")

            import plotly.graph_objects as go
            import plotly.express as px

            # Crear figura con barras coloreadas por Net GEX
            combined_fig = go.Figure()

            # Escala de color para Net GEX
            colorscale = px.colors.sequential.Viridis

            # Normalizar valores para usar en el color
            net_gex_normalized = (df["net_gex"] - df["net_gex"].min()) / (df["net_gex"].max() - df["net_gex"].min())

            combined_fig.add_bar(
                x=df["strike"],
                y=df["net_gex"],
                marker=dict(
                    color=net_gex_normalized,
                    colorscale=colorscale,
                    showscale=True,
                    colorbar=dict(title="Net GEX")
                ),
                name="Net GEX",
                hovertemplate="<b>Strike:</b> %{x}<br><b>Net GEX:</b> %{y:,.0f}<extra></extra>"
            )

            # Línea del Spot
            combined_fig.add_vline(
                x=spot,
                line_color="yellow",
                line_dash="dash",
                annotation_text="Spot",
                annotation_position="top"
            )

            # Anotaciones de strikes con OI alto
            top_oi_calls = calls.nlargest(3, "openInterest")
            top_oi_puts = puts.nlargest(3, "openInterest")
            for _, row in pd.concat([top_oi_calls, top_oi_puts]).iterrows():
                combined_fig.add_annotation(
                    x=row["strike"],
                    y=df.loc[df["strike"] == row["strike"], "net_gex"].values[0],
                    text=f"{int(row['strike'])}<br>OI {int(row['openInterest'])}",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-40,
                    bgcolor="rgba(0,0,0,0.5)",
                    font=dict(size=11, color="white")
                )

            combined_fig.update_layout(
                template="plotly_dark",
                title=f"OI vs Net GEX {ticker.upper()}",
                height=700,
                plot_bgcolor="#0e1621",
                paper_bgcolor="#0e1621",
                xaxis_title="Strike",
                yaxis_title="Net GEX",
                bargap=0.15,
                font=dict(size=12),
                showlegend=False
            )

            st.plotly_chart(combined_fig, width="stretch")

            # --- Análisis de Volatilidad (Smile y Term Structure) ---
            st.subheader("📈 Análisis de Volatilidad")
            import plotly.graph_objects as go
            # --- Volatility Smile ---
            vol_smile_fig = go.Figure()
            # Calls Smile
            vol_smile_fig.add_trace(go.Scatter(
                x=calls["strike"],
                y=calls["impliedVolatility"],
                mode="lines+markers",
                name="Calls",
                line=dict(color="yellow", width=2),
                marker=dict(color="yellow", size=6, symbol="circle"),
                hovertemplate="Strike: %{x}<br>IV: %{y:.2%}<extra></extra>"
            ))
            # Puts Smile
            vol_smile_fig.add_trace(go.Scatter(
                x=puts["strike"],
                y=puts["impliedVolatility"],
                mode="lines+markers",
                name="Puts",
                line=dict(color="#1f77b4", width=2),
                marker=dict(color="#1f77b4", size=6, symbol="diamond"),
                hovertemplate="Strike: %{x}<br>IV: %{y:.2%}<extra></extra>"
            ))
            vol_smile_fig.add_vline(
                x=spot,
                line_color="white",
                line_dash="dash",
                annotation_text=f"Spot {spot:.2f}",
                annotation_position="top right"
            )
            vol_smile_fig.update_layout(
                template="plotly_dark",
                title="Volatility Smile (IV vs Strike)",
                height=500,
                plot_bgcolor="#0e1621",
                paper_bgcolor="#0e1621",
                font=dict(size=12),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis_title="Strike",
                yaxis_title="Implied Volatility"
            )

            # --- Term Structure ---
            # (media de IV por expiración usando yf_get_expirations)
            exp_list = yf_get_expirations(ticker)
            term_data = []
            for exp in exp_list:
                c, p = yf_get_calls_puts_for_exp(ticker, exp)
                if c is not None and not c.empty and "impliedVolatility" in c.columns:
                    call_iv = c["impliedVolatility"].mean()
                else:
                    call_iv = None
                if p is not None and not p.empty and "impliedVolatility" in p.columns:
                    put_iv = p["impliedVolatility"].mean()
                else:
                    put_iv = None
                term_data.append({
                    "expiration": exp,
                    "call_iv": call_iv,
                    "put_iv": put_iv
                })
            term_df = pd.DataFrame(term_data)
            # Convertir expiración a fecha
            term_df["expiration"] = pd.to_datetime(term_df["expiration"])
            # Term structure plot
            term_struct_fig = go.Figure()
            term_struct_fig.add_trace(go.Scatter(
                x=term_df["expiration"],
                y=term_df["call_iv"],
                mode="lines+markers",
                name="Calls",
                line=dict(color="yellow", width=2),
                marker=dict(color="yellow", size=7, symbol="circle"),
                hovertemplate="Exp: %{x|%Y-%m-%d}<br>IV: %{y:.2%}<extra></extra>"
            ))
            term_struct_fig.add_trace(go.Scatter(
                x=term_df["expiration"],
                y=term_df["put_iv"],
                mode="lines+markers",
                name="Puts",
                line=dict(color="#1f77b4", width=2),
                marker=dict(color="#1f77b4", size=7, symbol="diamond"),
                hovertemplate="Exp: %{x|%Y-%m-%d}<br>IV: %{y:.2%}<extra></extra>"
            ))
            term_struct_fig.update_layout(
                template="plotly_dark",
                title="Term Structure de Volatilidad Implícita",
                height=450,
                plot_bgcolor="#0e1621",
                paper_bgcolor="#0e1621",
                font=dict(size=12),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis_title="Expiración",
                yaxis_title="Implied Volatility"
            )

            st.plotly_chart(vol_smile_fig, width="stretch")
            st.plotly_chart(term_struct_fig, width="stretch")

            # --- Etapa 4 — Integración cuantitativa ---
            # =======================================================
            st.subheader("🏦 Integración cuantitativa (Institucional)")
            import plotly.graph_objects as go
            import numpy as np
            # 1. IV Rank y IV Percentile (último año)
            try:
                hist_opt = yf.Ticker(ticker).option_chain(selected_exp)
                iv_mean = calls["impliedVolatility"].mean()
            except Exception:
                iv_mean = float("nan")
            try:
                # Obtener todos los expirations del último año, tomar el promedio de IV por cada uno
                hist_iv = []
                exp_list_1y = yf_get_expirations(ticker)
                for exp in exp_list_1y:
                    try:
                        c, p = yf_get_calls_puts_for_exp(ticker, exp)
                        if c is not None and not c.empty and "impliedVolatility" in c.columns:
                            hist_iv.append(c["impliedVolatility"].mean())
                    except Exception:
                        continue
                hist_iv = [iv for iv in hist_iv if iv is not None and not np.isnan(iv)]
                if len(hist_iv) > 0 and not np.isnan(iv_mean):
                    iv_rank = 100.0 * (iv_mean - np.min(hist_iv)) / (np.max(hist_iv) - np.min(hist_iv)) if np.max(hist_iv) != np.min(hist_iv) else 0.0
                    iv_percentile = 100.0 * (np.sum(np.array(hist_iv) < iv_mean) / len(hist_iv))
                else:
                    iv_rank = float("nan")
                    iv_percentile = float("nan")
            except Exception:
                iv_rank = float("nan")
                iv_percentile = float("nan")
            # 2. Historical Volatility (HV) comparada con IV actual
            try:
                hist_prices = yf.Ticker(ticker).history(period="1y")["Close"]
                log_ret = np.log(hist_prices / hist_prices.shift(1)).dropna()
                hv_daily = log_ret.std()
                hv_annualized = hv_daily * np.sqrt(252) * 100
            except Exception:
                hv_annualized = float("nan")
            # 3. Delta Notional agregado (abs)
            try:
                abs_delta_notional = np.abs(df["net_dex"]).sum()
            except Exception:
                abs_delta_notional = float("nan")
            # 4. Heatmap 3D GEX vs DTE
            try:
                # Para el heatmap 3D, necesitamos strikes, dte, net_gex para cada expiración
                # Recolectar para cada expiración (limitado a 10 para velocidad)
                heatmap_data = []
                exp_list_heatmap = exp_list_1y[:10] if len(exp_list_1y) > 10 else exp_list_1y
                for exp in exp_list_heatmap:
                    try:
                        c, p = yf_get_calls_puts_for_exp(ticker, exp)
                        spot_hm = spot
                        dte_hm = (pd.to_datetime(exp) - pd.Timestamp.today()).days
                        if c is not None and not c.empty and p is not None and not p.empty:
                            df_hm = compute_exposures(c, p, spot_hm, dte_hm, DEFAULT_R, DEFAULT_Q)
                            for i, row in df_hm.iterrows():
                                heatmap_data.append({
                                    "strike": row["strike"],
                                    "dte": dte_hm,
                                    "net_gex": row["net_gex"]
                                })
                    except Exception:
                        continue
                heatmap_df = pd.DataFrame(heatmap_data)
                if not heatmap_df.empty:
                    # Normalizar color por intensidad absoluta de net_gex
                    color_vals = np.abs(heatmap_df["net_gex"])
                    colorscale = "Viridis"
                    heatmap_3d_fig = go.Figure(data=[
                        go.Scatter3d(
                            x=heatmap_df["strike"],
                            y=heatmap_df["dte"],
                            z=heatmap_df["net_gex"],
                            mode="markers",
                            marker=dict(
                                size=5,
                                color=color_vals,
                                colorscale=colorscale,
                                colorbar=dict(title="|Net GEX|")
                            ),
                            text=[f"Strike: {s}<br>DTE: {d}<br>Net GEX: {g:,.0f}" for s, d, g in zip(heatmap_df["strike"], heatmap_df["dte"], heatmap_df["net_gex"])]
                        )
                    ])
                    heatmap_3d_fig.update_layout(
                        template="plotly_dark",
                        height=520,
                        title="3D GEX Heatmap (Strike vs DTE vs Net GEX)",
                        scene=dict(
                            xaxis_title="Strike",
                            yaxis_title="DTE (días)",
                            zaxis_title="Net GEX"
                        ),
                        plot_bgcolor="#0e1621",
                        paper_bgcolor="#0e1621",
                        font=dict(size=12)
                    )
                else:
                    heatmap_3d_fig = go.Figure()
                    heatmap_3d_fig.update_layout(
                        template="plotly_dark",
                        height=400,
                        title="No hay datos suficientes para el heatmap 3D",
                        plot_bgcolor="#0e1621",
                        paper_bgcolor="#0e1621"
                    )
            except Exception:
                heatmap_3d_fig = go.Figure()
                heatmap_3d_fig.update_layout(
                    template="plotly_dark",
                    height=400,
                    title="No hay datos suficientes para el heatmap 3D",
                    plot_bgcolor="#0e1621",
                    paper_bgcolor="#0e1621"
                )
            # Mostrar métricas en Streamlit
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("IV Rank (1y)", f"{iv_rank:.2f}%")
            with col2:
                st.metric("IV Percentile (1y)", f"{iv_percentile:.2f}%")
            with col3:
                st.metric("Historical Volatility", f"{hv_annualized:.2f}%")
            with col4:
                st.metric("Delta Notional (abs)", f"${abs_delta_notional/1e6:.2f}M")
            st.plotly_chart(heatmap_3d_fig, width="stretch")

            # --- Etapa 3 — Flujos dinámicos y Decay ---
            # =======================================================
            st.subheader("🔁 Flujos dinámicos y Decay")

            import numpy as np
            import plotly.graph_objects as go

            # Simular OI anterior para ilustración (en producción, cargar histórico real)
            np.random.seed(42)
            calls_oi_prev = calls["openInterest"] + np.random.randint(-5, 6, size=len(calls))
            puts_oi_prev = puts["openInterest"] + np.random.randint(-5, 6, size=len(puts))
            # Clamp para que no sea negativo
            calls_oi_prev = np.maximum(0, calls_oi_prev)
            puts_oi_prev = np.maximum(0, puts_oi_prev)

            # Calcular variación diaria de OI (ΔOI)
            calls_delta_oi = calls["openInterest"] - calls_oi_prev
            puts_delta_oi = puts["openInterest"] - puts_oi_prev

            # Gráfico de ΔOI
            delta_oi_fig = go.Figure()
            # Calls: verde (positivo), rojo (negativo)
            delta_oi_fig.add_bar(
                x=calls["strike"],
                y=calls_delta_oi,
                name="ΔOI Calls",
                marker_color=["#2ecc40" if v >= 0 else "#ff4136" for v in calls_delta_oi]
            )
            # Puts: verde (positivo), rojo (negativo), invertidos para visual
            delta_oi_fig.add_bar(
                x=puts["strike"],
                y=-puts_delta_oi,
                name="ΔOI Puts",
                marker_color=["#2ecc40" if v >= 0 else "#ff4136" for v in puts_delta_oi]
            )
            delta_oi_fig.add_vline(
                x=spot,
                line_color="yellow",
                line_dash="dash",
                annotation_text=f"Spot {spot:.2f}",
                annotation_position="top right"
            )
            delta_oi_fig.update_layout(
                template="plotly_dark",
                barmode="overlay",
                height=400,
                bargap=0.18,
                title="Variación diaria de Open Interest (ΔOI)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                font=dict(size=12),
                plot_bgcolor="#0e1621",
                paper_bgcolor="#0e1621",
                yaxis_title="ΔOI (positivo = aumento, negativo = reducción)",
                xaxis_title="Strike"
            )

            # --- Calcular Theta Exposure (THX) y Vega Exposure (VEX) por strike ---
            # Black-Scholes formulas
            def bs_d1(S, K, T, r, q, sigma):
                return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            def bs_d2(S, K, T, r, q, sigma):
                return bs_d1(S, K, T, r, q, sigma) - sigma * np.sqrt(T)
            def bs_theta_call(S, K, T, r, q, sigma):
                d1 = bs_d1(S, K, T, r, q, sigma)
                d2 = bs_d2(S, K, T, r, q, sigma)
                return (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                        - r * K * np.exp(-r * T) * norm.cdf(d2)
                        + q * S * np.exp(-q * T) * norm.cdf(d1))
            def bs_theta_put(S, K, T, r, q, sigma):
                d1 = bs_d1(S, K, T, r, q, sigma)
                d2 = bs_d2(S, K, T, r, q, sigma)
                return (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                        + r * K * np.exp(-r * T) * norm.cdf(-d2)
                        - q * S * np.exp(-q * T) * norm.cdf(-d1))
            def bs_vega(S, K, T, r, q, sigma):
                d1 = bs_d1(S, K, T, r, q, sigma)
                return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

            # T (years) para la expiración seleccionada
            dte = (pd.to_datetime(selected_exp) - pd.Timestamp.today()).days
            T = max(dte, 1) / 365
            r = DEFAULT_R
            q = DEFAULT_Q

            # Calcular THX y VEX por strike
            calls_sigma = calls["impliedVolatility"].fillna(0.3)
            puts_sigma = puts["impliedVolatility"].fillna(0.3)
            calls_thx = bs_theta_call(spot, calls["strike"], T, r, q, calls_sigma) * 100 * calls["openInterest"]
            puts_thx = bs_theta_put(spot, puts["strike"], T, r, q, puts_sigma) * 100 * puts["openInterest"]
            calls_vex = bs_vega(spot, calls["strike"], T, r, q, calls_sigma) * 100 * calls["openInterest"]
            puts_vex = bs_vega(spot, puts["strike"], T, r, q, puts_sigma) * 100 * puts["openInterest"]

            # Sumar por strike (por si hay strikes repetidos)
            thx_df = pd.DataFrame({
                "strike": pd.concat([calls["strike"], puts["strike"]]),
                "thx": pd.concat([calls_thx, puts_thx]),
                "vex": pd.concat([calls_vex, puts_vex])
            })
            thx_agg = thx_df.groupby("strike", as_index=False).sum()

            # Gráfico combinado THX/VEX
            thx_vex_fig = go.Figure()
            thx_vex_fig.add_trace(go.Scatter(
                x=thx_agg["strike"],
                y=thx_agg["thx"],
                mode="lines+markers",
                name="THX (Theta Exposure)",
                line=dict(color="mediumpurple", width=2),
                marker=dict(size=6)
            ))
            thx_vex_fig.add_trace(go.Scatter(
                x=thx_agg["strike"],
                y=thx_agg["vex"],
                mode="lines+markers",
                name="VEX (Vega Exposure)",
                line=dict(color="cyan", width=2),
                marker=dict(size=6)
            ))
            thx_vex_fig.add_vline(
                x=spot,
                line_color="yellow",
                line_dash="dash",
                annotation_text=f"Spot {spot:.2f}",
                annotation_position="top right"
            )
            thx_vex_fig.update_layout(
                template="plotly_dark",
                title="Theta & Vega Exposure agregados por Strike",
                height=430,
                plot_bgcolor="#0e1621",
                paper_bgcolor="#0e1621",
                font=dict(size=12),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis_title="Strike",
                yaxis_title="Exposure (USD, sumado Calls+Puts)"
            )

            st.plotly_chart(delta_oi_fig, width="stretch")
            st.plotly_chart(thx_vex_fig, width="stretch")

            # --- Gamma Flip & Dealer Bias ---
            st.subheader("⚖️ Gamma Flip & Dealer Bias")

            def compute_gamma_flip(df, spot):
                import numpy as np
                if df is None or df.empty or "net_gex" not in df.columns or "strike" not in df.columns:
                    return None, "N/A"
                strikes = df["strike"].values
                net_gex = df["net_gex"].values
                # Buscar cruces de signo en net_gex
                sign_changes = np.where(np.diff(np.sign(net_gex)) != 0)[0]
                if len(sign_changes) == 0:
                    # No hay cruce de gamma
                    closest_idx = np.abs(net_gex).argmin()
                    gamma_flip = strikes[closest_idx]
                else:
                    # Tomar el primer cruce de signo
                    idx = sign_changes[0]
                    # Interpolación lineal para estimar el cruce exacto
                    x0, x1 = strikes[idx], strikes[idx + 1]
                    y0, y1 = net_gex[idx], net_gex[idx + 1]
                    if y1 - y0 != 0:
                        gamma_flip = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
                    else:
                        gamma_flip = x0
                # Dealer bias
                total_net_gex = np.nansum(net_gex)
                dealer_bias = "Long Gamma" if total_net_gex > 0 else "Short Gamma"
                return gamma_flip, dealer_bias

            gamma_flip, dealer_bias = compute_gamma_flip(df, spot)
            col1, col2 = st.columns(2)
            with col1:
                if gamma_flip is not None:
                    st.metric("Gamma Flip Strike", f"${gamma_flip:.2f}")
                else:
                    st.metric("Gamma Flip Strike", "N/A")
            with col2:
                st.metric("Dealer Bias", dealer_bias)

            # Calcular métricas adicionales
            iv_mean = calls["impliedVolatility"].mean()
            dte = (pd.to_datetime(selected_exp) - pd.Timestamp.today()).days
            mmm = spot * iv_mean * np.sqrt(dte / 365)
            mmm_pct = (mmm / spot) * 100
            upper_band = spot + mmm
            lower_band = spot - mmm

            # Crear tabla de resumen con valores completos
            summary_df = pd.DataFrame([{
                "Weekly": selected_exp,
                "Mid Month": None,
                "EOM": None,
                "Implied Vol (local)": f"{iv_mean:.2f}",
                "DTE": dte,
                "MMM": f"{mmm:.2f}",
                "%MMM": f"{mmm_pct:.2f}%",
                "Upper Band$": f"{upper_band:.2f}",
                "Lower Band$": f"{lower_band:.2f}"
            }])

            st.subheader("📋 Tabla de resumen")
            st.dataframe(summary_df, width="stretch")

            # =======================================================
            #  --- ESCÁNER DE OPCIONES ---
            # =======================================================
            st.subheader("📈 Escáner de opciones")

            def find_nearest(df, col, value, greater=True):
                if greater:
                    filtered = df[df[col] > value]
                    return filtered.iloc[(filtered[col] - value).abs().argsort()[:1]] if not filtered.empty else pd.DataFrame()
                else:
                    filtered = df[df[col] < value]
                    return filtered.iloc[(filtered[col] - value).abs().argsort()[:1]] if not filtered.empty else pd.DataFrame()

            # Calls: mayor OTM y menor ITM
            call_otm = find_nearest(calls, "strike", spot, greater=True)
            call_itm = find_nearest(calls, "strike", spot, greater=False)

            # Puts: mayor ITM (strike > spot) y menor OTM (strike < spot)
            put_itm = find_nearest(puts, "strike", spot, greater=True)
            put_otm = find_nearest(puts, "strike", spot, greater=False)

            # Combinar resultados
            scanner_rows = []
            if not call_otm.empty:
                scanner_rows.append({
                    "Descripción": "Mayor OTM Call",
                    "Strike": f"{call_otm['strike'].iloc[0]:.2f}",
                    "Delta": f"{call_otm['delta'].iloc[0]:.2f}",
                    "Días": dte,
                    "% Cambio": f"{call_otm['change'].iloc[0]:.2f}%",
                    "Volat Impl": f"{call_otm['impliedVolatility'].iloc[0]*100:.2f}%"
                })
            if not call_itm.empty:
                scanner_rows.append({
                    "Descripción": "Menor ITM Call",
                    "Strike": f"{call_itm['strike'].iloc[0]:.2f}",
                    "Delta": f"{call_itm['delta'].iloc[0]:.2f}",
                    "Días": dte,
                    "% Cambio": f"{call_itm['change'].iloc[0]:.2f}%",
                    "Volat Impl": f"{call_itm['impliedVolatility'].iloc[0]*100:.2f}%"
                })
            if not put_itm.empty:
                scanner_rows.append({
                    "Descripción": "Mayor ITM Put",
                    "Strike": f"{put_itm['strike'].iloc[0]:.2f}",
                    "Delta": f"{put_itm['delta'].iloc[0]:.2f}",
                    "Días": dte,
                    "% Cambio": f"{put_itm['change'].iloc[0]:.2f}%",
                    "Volat Impl": f"{put_itm['impliedVolatility'].iloc[0]*100:.2f}%"
                })
            if not put_otm.empty:
                scanner_rows.append({
                    "Descripción": "Menor OTM Put",
                    "Strike": f"{put_otm['strike'].iloc[0]:.2f}",
                    "Delta": f"{put_otm['delta'].iloc[0]:.2f}",
                    "Días": dte,
                    "% Cambio": f"{put_otm['change'].iloc[0]:.2f}%",
                    "Volat Impl": f"{put_otm['impliedVolatility'].iloc[0]*100:.2f}%"
                })

            if scanner_rows:
                scanner_df = pd.DataFrame(scanner_rows)
                st.dataframe(scanner_df, width="stretch")
            else:
                st.info("No se pudieron generar datos para el escáner de opciones.")
