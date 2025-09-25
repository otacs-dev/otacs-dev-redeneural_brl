import os
from pathlib import Path
from math import erf, sqrt
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_collector import DataCollector
from src.technical_analysis import TechnicalAnalysis
from src.visualizations import Visualizations


APP_DIR = Path(__file__).resolve().parent
if Path.cwd() != APP_DIR:
    os.chdir(APP_DIR)

st.set_page_config(
    page_title="Painel Inteligente de Ações BRL",
    page_icon="📈",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            body {
                background-color: #f6f8fb;
                color: #1f2933;
                font-family: "Inter", "Segoe UI", sans-serif;
            }

            .stApp header {display: none;}

            .dashboard-card {
                background: #ffffff;
                border-radius: 18px;
                padding: 1.4rem 1.6rem;
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
                border: 1px solid rgba(15, 23, 42, 0.04);
                margin-bottom: 1.5rem;
            }

            .metric-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1.25rem;
            }

            .metric-chip {
                display: flex;
                flex-direction: column;
                gap: 0.4rem;
                padding: 1rem 1.2rem;
                background: #f8fafc;
                border-radius: 14px;
                border: 1px solid rgba(148, 163, 184, 0.25);
            }

            .metric-chip h4 {
                font-size: 0.85rem;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin: 0;
            }

            .metric-chip span {
                font-size: 1.4rem;
                font-weight: 600;
                color: #0f172a;
            }

            .tag {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.35rem 0.75rem;
                border-radius: 999px;
                font-size: 0.85rem;
                font-weight: 600;
            }

            .tag.positive {background: rgba(34, 197, 94, 0.12); color: #047857;}
            .tag.neutral {background: rgba(59, 130, 246, 0.12); color: #1d4ed8;}
            .tag.negative {background: rgba(248, 113, 113, 0.12); color: #b91c1c;}

            .recommendation-card {
                border-radius: 18px;
                padding: 1.6rem;
                color: #0f172a;
                display: flex;
                flex-direction: column;
                gap: 0.8rem;
            }

            .recommendation-card h2 {
                margin: 0;
                font-size: 2.2rem;
                font-weight: 700;
            }

            .confidence-bar {
                width: 100%;
                height: 10px;
                border-radius: 999px;
                background: rgba(148, 163, 184, 0.2);
                overflow: hidden;
                margin-top: 0.35rem;
            }

            .confidence-bar span {
                display: block;
                height: 100%;
                border-radius: inherit;
            }

            .event-card {
                background: #ffffff;
                border-radius: 14px;
                padding: 1rem 1.2rem;
                border: 1px solid rgba(148, 163, 184, 0.18);
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
                box-shadow: 0 12px 24px rgba(15, 23, 42, 0.04);
            }

            .event-card h4 {
                margin: 0;
                font-size: 1rem;
                font-weight: 600;
            }

            .event-date {color: #64748b; font-size: 0.85rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_currency(value: float) -> str:
    if pd.isna(value):
        return "N/D"
    return f"R$ {value:,.2f}".replace(",", "@").replace(".", ",").replace("@", ".")


def format_percentage(value: float) -> str:
    if pd.isna(value):
        return "N/D"
    return f"{value:.2f}%"


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def calculate_risk_metrics(data: pd.DataFrame) -> dict:
    returns = data["Returns"].dropna()

    if len(returns) < 30:
        return {
            "volatility": float("nan"),
            "var_95": float("nan"),
            "max_drawdown": float("nan"),
            "sharpe_ratio": float("nan"),
            "downside_risk": float("nan"),
            "sortino_ratio": float("nan"),
        }

    volatility = returns.std() * np.sqrt(252) * 100
    var_95 = np.percentile(returns, 5) * 100

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max - 1) * 100
    max_drawdown = drawdown.min()

    risk_free = 0.05 / 252
    sharpe = ((returns.mean() - risk_free) / (returns.std() + 1e-9)) * np.sqrt(252)

    negative_returns = returns[returns < 0]
    downside_vol = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else np.nan
    sortino = ((returns.mean() - risk_free) / (downside_vol + 1e-9)) * np.sqrt(252)

    return {
        "volatility": volatility,
        "var_95": var_95,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "downside_risk": downside_vol * 100 if not np.isnan(downside_vol) else float("nan"),
        "sortino_ratio": sortino,
    }


def compute_company_health(data: pd.DataFrame) -> dict:
    window = min(len(data), 252)
    if window < 60:
        return {
            "score": 50,
            "growth_score": 50,
            "profitability_score": 50,
            "stability_score": 50,
            "factors": {
                "growth": "Limitado",
                "profitability": "Neutro",
                "stability": "Moderada",
                "efficiency": "Neutra",
            },
        }

    recent = data.tail(window)
    prices = recent["Close"]
    returns = prices.pct_change().dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    annual_return = (prices.iloc[-1] / prices.iloc[0]) ** (252 / window) - 1
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
    volatility = returns.std() * np.sqrt(252)
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    max_drawdown = (cumulative / running_max - 1).min()

    growth_score = np.interp(annual_return, [-0.4, 0.0, 0.4], [10, 50, 90])
    profitability_score = np.interp(sharpe, [-1.0, 0.0, 2.0], [15, 55, 95])
    stability_score = np.interp(-max_drawdown, [0.05, 0.25, 0.6], [90, 55, 20])
    efficiency_score = np.interp(volatility, [0.12, 0.30, 0.55], [90, 55, 20])

    overall = 0.4 * growth_score + 0.3 * profitability_score + 0.2 * stability_score + 0.1 * efficiency_score

    def label(score: float) -> str:
        if score >= 70:
            return "Forte"
        if score >= 50:
            return "Moderado"
        return "Fraco"

    return {
        "score": round(float(np.clip(overall, 0, 100))),
        "growth_score": float(growth_score),
        "profitability_score": float(profitability_score),
        "stability_score": float(stability_score),
        "factors": {
            "growth": label(growth_score),
            "profitability": label(profitability_score),
            "stability": label(stability_score),
            "efficiency": label(efficiency_score),
        },
        "annual_return": annual_return * 100,
        "volatility": volatility * 100,
        "max_drawdown": max_drawdown * 100,
        "sharpe": sharpe,
    }


def compute_market_sentiment(data: pd.DataFrame) -> dict:
    returns = data["Returns"].dropna()
    if len(returns) < 20:
        return {
            "score": 0.0,
            "rating": "Neutro",
            "momentum": 0.0,
            "volume_trend": 0.0,
            "breadth": 0.0,
        }

    recent_returns = returns.tail(30)
    if recent_returns.empty:
        recent_returns = returns

    momentum = recent_returns.mean() * 252
    breadth = (recent_returns > 0).mean() - (recent_returns < 0).mean()

    volume = data["Volume"].tail(30)
    long_volume = data["Volume"].tail(90)
    volume_trend = (volume.mean() / long_volume.mean() - 1) if not long_volume.empty else 0.0

    raw_score = 0.6 * momentum + 0.3 * breadth + 0.1 * volume_trend
    score = float(np.tanh(raw_score))

    if score > 0.25:
        rating = "Positivo"
    elif score < -0.25:
        rating = "Negativo"
    else:
        rating = "Neutro"

    return {
        "score": score,
        "rating": rating,
        "momentum": momentum * 100,
        "volume_trend": volume_trend * 100,
        "breadth": breadth * 100,
    }


def extract_price_events(data: pd.DataFrame, window: int = 160) -> list:
    if len(data) < 40:
        return []

    recent = data.tail(window).copy()
    returns = recent["Returns"].dropna()
    if returns.empty:
        return []

    threshold = returns.std() * 2
    events = []

    for date, ret in returns.iloc[::-1].items():
        if abs(ret) < threshold:
            continue

        direction = "Alta" if ret > 0 else "Queda"
        magnitude = ret * 100
        avg_volume = recent["Volume"].rolling(20).mean().get(date, np.nan)
        volume_ratio = recent.loc[date, "Volume"] / avg_volume if avg_volume and not np.isnan(avg_volume) else 1.0

        events.append(
            {
                "date": date,
                "direction": direction,
                "magnitude": magnitude,
                "volume_ratio": volume_ratio,
                "description": f"Movimento de {magnitude:.2f}% com volume {volume_ratio:.1f}x a média",
            }
        )

        if len(events) >= 6:
            break

    return events


def project_price_distribution(data: pd.DataFrame, horizon_days: int) -> dict | None:
    log_returns = data.get("Log_Returns")
    if log_returns is None:
        log_returns = np.log(data["Close"] / data["Close"].shift(1))

    log_returns = log_returns.dropna()
    if len(log_returns) < 20:
        return None

    current_price = data["Close"].iloc[-1]
    drift = log_returns.mean()
    volatility = log_returns.std()

    mean_ln = np.log(current_price) + drift * horizon_days
    std_ln = volatility * np.sqrt(horizon_days)

    predicted_price = float(np.exp(mean_ln))
    lower = float(np.exp(mean_ln - 1.96 * std_ln))
    upper = float(np.exp(mean_ln + 1.96 * std_ln))

    if std_ln == 0:
        prob_up = 0.5
    else:
        prob_up = float(normal_cdf((mean_ln - np.log(current_price)) / std_ln))

    direction_confidence = max(prob_up, 1 - prob_up) * 100
    expected_return = (predicted_price / current_price - 1) * 100

    return {
        "predicted_price": predicted_price,
        "lower_bound": lower,
        "upper_bound": upper,
        "expected_return_pct": expected_return,
        "prob_up": prob_up,
        "direction_confidence": direction_confidence,
        "std_ln": std_ln,
        "current_price": float(current_price),
        "horizon_days": horizon_days,
    }


def build_recommendation(forecast: dict, health: dict, sentiment: dict, data: pd.DataFrame) -> dict:
    sentiment_norm = (sentiment["score"] + 1) / 2 * 100
    expected_norm = np.clip((forecast["expected_return_pct"] + 12) / 24 * 100, 0, 100)

    trend_indicator = data.get("Trend_Indicator")
    if trend_indicator is not None and not np.isnan(trend_indicator.iloc[-1]):
        trend_norm = float(trend_indicator.iloc[-1] * 100)
    else:
        trend_norm = 50.0

    composite = 0.4 * expected_norm + 0.35 * health["score"] + 0.25 * (0.5 * sentiment_norm + 0.5 * trend_norm)
    composite = float(np.clip(composite, 0, 100))

    if composite >= 70:
        recommendation = "COMPRAR"
        color = "#0f766e"
    elif composite <= 35:
        recommendation = "VENDER"
        color = "#b91c1c"
    else:
        recommendation = "MANTER"
        color = "#d97706"

    confidence = min(95.0, 0.6 * forecast["direction_confidence"] + 0.4 * abs(sentiment["score"]) * 100)
    certainty = max(20.0, 90.0 - forecast["std_ln"] * 180)

    return {
        "score": composite,
        "recommendation": recommendation,
        "color": color,
        "confidence": confidence,
        "certainty": certainty,
    }


def build_forecasts(data: pd.DataFrame, horizons: dict[str, int], health: dict, sentiment: dict) -> dict:
    forecasts: dict[str, dict] = {}

    for label, days in horizons.items():
        projection = project_price_distribution(data, days)
        if projection is None:
            forecasts[label] = {}
            continue

        recommendation = build_recommendation(projection, health, sentiment, data)
        forecasts[label] = {**projection, **recommendation}

    return forecasts


def technical_snapshot(data: pd.DataFrame) -> pd.Series:
    latest = data.iloc[-1]
    summary = {
        "RSI": latest.get("RSI", np.nan),
        "MACD": latest.get("MACD", np.nan),
        "MACD - Sinal": latest.get("MACD", np.nan) - latest.get("MACD_Signal", np.nan),
        "Posição BB (%)": latest.get("BB_Position", np.nan) * 100,
        "Momento 10d (%)": latest.get("Momentum", np.nan) * 100,
        "Tendência (0-100)": latest.get("Trend_Indicator", np.nan) * 100,
        "Volume x Média": latest.get("Volume_Ratio", np.nan) * 100,
    }
    return pd.Series(summary)


def relative_performance(selected: str, base_data: pd.DataFrame, symbols: list[str], days: int, collector: DataCollector, offline: bool) -> pd.DataFrame:
    rows = []
    base_close = base_data["Close"]

    def compute_period_return(series: pd.Series, period: int) -> float:
        if len(series) <= period:
            return np.nan
        return (series.iloc[-1] / series.iloc[-period - 1] - 1) * 100

    for symbol in symbols:
        if symbol == selected:
            prices = base_close
        else:
            fetched = collector.get_stock_data(symbol, days=days, offline_mode=offline)
            if fetched is None or fetched.empty:
                continue
            prices = fetched["Close"]

        rows.append(
            {
                "Ticker": symbol.replace(".SA", ""),
                "30d": compute_period_return(prices, 21),
                "90d": compute_period_return(prices, 63),
                "180d": compute_period_return(prices, 126),
                "YTD": (prices.iloc[-1] / prices.iloc[0] - 1) * 100 if len(prices) > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows).set_index("Ticker")


def render_forecast_chart(data: pd.DataFrame, forecast: dict) -> go.Figure:
    current_date = data.index[-1]
    target_date = current_date + timedelta(days=int(forecast["horizon_days"]))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index[-120:],
            y=data["Close"].tail(120),
            mode="lines",
            name="Histórico",
            line=dict(color="#1d4ed8", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[current_date, target_date],
            y=[forecast["current_price"], forecast["predicted_price"]],
            mode="lines+markers",
            name="Projeção",
            line=dict(color="#0f766e", width=3, dash="dash"),
            marker=dict(size=8),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[target_date, target_date],
            y=[forecast["lower_bound"], forecast["upper_bound"]],
            fill=None,
            mode="lines",
            line=dict(color="rgba(15, 118, 110, 0.2)", width=0),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[target_date, target_date],
            y=[forecast["lower_bound"], forecast["upper_bound"]],
            fill="tonexty",
            mode="lines",
            line=dict(color="rgba(15, 118, 110, 0.2)", width=0),
            showlegend=True,
            name="Intervalo 95%",
        )
    )

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
        template="plotly_white",
        yaxis_title="Preço (R$)",
    )

    return fig


inject_styles()

st.title("📊 Painel Inteligente de Ações Brasileiras")
st.caption("Análise combinada de preço, risco e projeções probabilísticas para ativos listados na B3.")

with st.sidebar:
    st.header("Configurações")
    stocks = ["VIVA3.SA", "MOVI3.SA", "TRIS3.SA", "AMER3.SA", "MGLU3.SA"]
    stock_labels = {
        "VIVA3.SA": "Vivara (VIVA3)",
        "MOVI3.SA": "Movida (MOVI3)",
        "TRIS3.SA": "Trisul (TRIS3)",
        "AMER3.SA": "Americanas (AMER3)",
        "MGLU3.SA": "Magazine Luiza (MGLU3)",
    }

    selected_stock = st.selectbox("Escolha o ativo", stocks, format_func=lambda x: stock_labels[x])

    history_options = {
        "2 anos": 730,
        "3 anos": 1095,
        "5 anos": 1825,
    }
    history_label = st.select_slider("Histórico analisado", options=list(history_options.keys()), value="3 anos")
    history_days = history_options[history_label]

    use_offline = st.toggle("Usar dados armazenados (offline)", value=False)

    horizon_map = {
        "1 semana": 5,
        "1 mês": 21,
        "3 meses": 63,
        "6 meses": 126,
        "1 ano": 252,
    }
    horizon_label = st.selectbox("Horizonte detalhado", list(horizon_map.keys()), index=1)

collector = DataCollector()
analysis_engine = TechnicalAnalysis()
visuals = Visualizations()

with st.spinner(f"Carregando dados históricos de {stock_labels[selected_stock]}..."):
    stock_data = collector.get_stock_data(selected_stock, days=history_days, offline_mode=use_offline)

if stock_data is None or stock_data.empty:
    st.error("Não foi possível carregar os dados para este ativo. Verifique a conexão ou baixe os dados previamente.")
    st.stop()

price_data = analysis_engine.calculate_indicators(stock_data)

risk_metrics = calculate_risk_metrics(price_data)
health_info = compute_company_health(price_data)
sentiment_info = compute_market_sentiment(price_data)
price_events = extract_price_events(price_data)
forecasts = build_forecasts(price_data, horizon_map, health_info, sentiment_info)
selected_forecast = forecasts.get(horizon_label, {})

last_price = price_data["Close"].iloc[-1]
previous_price = price_data["Close"].iloc[-2] if len(price_data) > 1 else last_price
price_delta = (last_price - previous_price) / previous_price * 100 if previous_price != 0 else 0

fig_price = visuals.plot_candlestick_with_indicators(price_data, stock_labels[selected_stock])

tab_overview, tab_forecast, tab_details = st.tabs(["Visão Geral", "Previsões", "Indicadores & Risco"])

with tab_overview:
    col_chart, col_metrics = st.columns([2.5, 1])

    with col_chart:
        st.plotly_chart(fig_price, use_container_width=True)

    with col_metrics:
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("#### Panorama do Dia")
        st.metric("Preço atual", format_currency(last_price), format_percentage(price_delta))
        st.metric("Volatilidade anual", format_percentage(risk_metrics["volatility"]))
        st.metric("Volume do dia", f"{int(price_data['Volume'].iloc[-1]):,}".replace(",", "."))
        if use_offline:
            st.info("Modo offline ativado · dados provenientes do cache local")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("#### Sentimento de Mercado")
        sentiment_class = "positive" if sentiment_info["rating"] == "Positivo" else "negative" if sentiment_info["rating"] == "Negativo" else "neutral"
        st.markdown(
            f"<span class='tag {sentiment_class}'>Sentimento {sentiment_info['rating']} · {sentiment_info['score']*100:.1f}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='metric-grid'>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='metric-chip'><h4>Momentum anual</h4><span>{format_percentage(sentiment_info['momentum'])}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='metric-chip'><h4>Volume vs 90d</h4><span>{format_percentage(sentiment_info['volume_trend'])}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='metric-chip'><h4>Amplitude direcional</h4><span>{format_percentage(sentiment_info['breadth'])}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("#### Eventos recentes relevantes")
    if price_events:
        cols = st.columns(3)
        for idx, event in enumerate(price_events):
            with cols[idx % 3]:
                st.markdown("<div class='event-card'>", unsafe_allow_html=True)
                st.markdown(f"<span class='event-date'>{event['date'].strftime('%d/%m/%Y')}</span>", unsafe_allow_html=True)
                direction_class = "positive" if event["direction"] == "Alta" else "negative"
                st.markdown(
                    f"<span class='tag {direction_class}'>{event['direction']} · {event['magnitude']:.2f}%</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"<p>{event['description']}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Nenhum movimento extraordinário identificado no período analisado.")

with tab_forecast:
    st.markdown("#### Recomendação combinada")
    if selected_forecast:
        st.markdown(
            f"<div class='recommendation-card' style='background: {selected_forecast['color']}14'>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<h2 style='color:{selected_forecast['color']}'>{selected_forecast['recommendation']}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"Score composto: **{selected_forecast['score']:.1f} / 100**",
        )
        st.markdown(
            f"Preço projetado: **{format_currency(selected_forecast['predicted_price'])}**"
            f" ({format_percentage(selected_forecast['expected_return_pct'])})",
        )
        st.markdown(
            "Confiabilidade da direção prevista",
        )
        st.markdown("<div class='confidence-bar'>", unsafe_allow_html=True)
        st.markdown(
            f"<span style='width:{selected_forecast['confidence']:.1f}%; background:{selected_forecast['color']}'></span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p>Confiança: {selected_forecast['confidence']:.1f}% · Certeza estatística: {selected_forecast['certainty']:.1f}%</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        forecast_chart = render_forecast_chart(price_data, selected_forecast)
        st.plotly_chart(forecast_chart, use_container_width=True)
    else:
        st.warning("Dados insuficientes para calcular a projeção selecionada.")

    table_rows = []
    for label, info in forecasts.items():
        if not info:
            continue
        table_rows.append(
            {
                "Horizonte": label,
                "Preço projetado": info["predicted_price"],
                "Retorno esperado (%)": info["expected_return_pct"],
                "Prob. de alta (%)": info["prob_up"] * 100,
                "Confiança (%)": info["confidence"],
                "Recomendação": info["recommendation"],
            }
        )

    if table_rows:
        forecast_df = pd.DataFrame(table_rows).set_index("Horizonte")
        st.dataframe(
            forecast_df.style.format(
                {
                    "Preço projetado": format_currency,
                    "Retorno esperado (%)": "{:.2f}",
                    "Prob. de alta (%)": "{:.1f}",
                    "Confiança (%)": "{:.1f}",
                }
            ),
            use_container_width=True,
        )
    else:
        st.info("As projeções probabilísticas ficarão disponíveis assim que houver histórico suficiente.")

with tab_details:
    st.markdown("#### Indicadores técnicos atuais")
    indicators = technical_snapshot(price_data)
    st.dataframe(indicators.to_frame(name="Valor").style.format("{:.2f}"), use_container_width=True)

    st.markdown("#### Métricas de risco")
    risk_df = pd.DataFrame(
        {
            "Métrica": [
                "Volatilidade anual",
                "VaR 95%",
                "Máximo drawdown",
                "Sharpe",
                "Risco de queda",
                "Sortino",
            ],
            "Valor": [
                risk_metrics["volatility"],
                risk_metrics["var_95"],
                risk_metrics["max_drawdown"],
                risk_metrics["sharpe_ratio"],
                risk_metrics["downside_risk"],
                risk_metrics["sortino_ratio"],
            ],
        }
    )
    st.dataframe(
        risk_df.set_index("Métrica").style.format(
            {
                "Valor": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

    st.markdown("#### Saúde financeira estimada a partir do preço")
    health_df = pd.DataFrame(
        {
            "Indicador": ["Score geral", "Crescimento", "Lucratividade", "Estabilidade", "Eficiência"],
            "Valor": [
                health_info["score"],
                health_info["factors"]["growth"],
                health_info["factors"]["profitability"],
                health_info["factors"]["stability"],
                health_info["factors"]["efficiency"],
            ],
        }
    )
    st.dataframe(health_df.set_index("Indicador"), use_container_width=True)

    st.markdown("#### Desempenho relativo na B3")
    performance_df = relative_performance(selected_stock, price_data, stocks, history_days, collector, use_offline)
    if not performance_df.empty:
        st.dataframe(
            performance_df.style.format("{:.2f}"),
            use_container_width=True,
        )
    else:
        st.info("Não foi possível comparar o desempenho com outros ativos no momento.")


