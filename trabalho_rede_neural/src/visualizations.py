import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('stock_prediction_app.visualizations')

class Visualizations:
    """
    Classe para geração de visualizações interativas dos dados e previsões.
    """
    
    def __init__(self):
        # Definir cores para melhor visualização
        self.up_color = '#26a69a'
        self.down_color = '#ef5350'
        self.neutral_color = '#1976d2'
        self.band_color = 'rgba(0, 0, 0, 0.1)'
    
    def plot_candlestick_with_indicators(self, data, title):
        """
        Cria um gráfico de candlestick com indicadores técnicos.
        
        Args:
            data: DataFrame com dados históricos e indicadores
            title: Título do gráfico
            
        Returns:
            Figura plotly
        """
        try:
            # Obter dados mais recentes (último ano)
            recent_data = data.iloc[-252:].copy() if len(data) > 252 else data.copy()
            
            # Criar subplots: preço e volume
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=(f"{title} - Preço e Indicadores", "Volume")
            )
            
            # Adicionar candlestick
            fig.add_trace(
                go.Candlestick(
                    x=recent_data.index,
                    open=recent_data['Open'],
                    high=recent_data['High'],
                    low=recent_data['Low'],
                    close=recent_data['Close'],
                    name="Preço",
                    increasing_line_color=self.up_color,
                    decreasing_line_color=self.down_color
                ),
                row=1, col=1
            )
            
            # Adicionar médias móveis
            fig.add_trace(
                go.Scatter(
                    x=recent_data.index,
                    y=recent_data['SMA_20'],
                    name="SMA 20",
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=recent_data.index,
                    y=recent_data['SMA_50'],
                    name="SMA 50",
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
            
            # Adicionar Bandas de Bollinger
            fig.add_trace(
                go.Scatter(
                    x=recent_data.index,
                    y=recent_data['BB_Upper'],
                    name="BB Superior",
                    line=dict(color='rgba(0, 0, 0, 0.3)', width=1),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=recent_data.index,
                    y=recent_data['BB_Lower'],
                    name="BB Inferior",
                    line=dict(color='rgba(0, 0, 0, 0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(0, 0, 0, 0.05)',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Adicionar volumes
            colors = [self.up_color if row['Close'] >= row['Open'] else self.down_color 
                      for _, row in recent_data.iterrows()]
            
            fig.add_trace(
                go.Bar(
                    x=recent_data.index,
                    y=recent_data['Volume'],
                    name="Volume",
                    marker_color=colors
                ),
                row=2, col=1
            )
            
            # Adicionar linha de média de volume
            fig.add_trace(
                go.Scatter(
                    x=recent_data.index,
                    y=recent_data['Volume_SMA'],
                    name="Volume Médio",
                    line=dict(color='darkgray', width=1)
                ),
                row=2, col=1
            )
            
            # Layout e configurações
            fig.update_layout(
                title=f"{title}",
                xaxis_title="Data",
                yaxis_title="Preço (R$)",
                height=600,
                margin=dict(l=50, r=50, t=50, b=50),
                xaxis_rangeslider_visible=False,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                template="plotly_white"
            )
            
            # Configurar eixos
            fig.update_yaxes(title_text="Preço (R$)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro ao criar gráfico: {str(e)}")
            
            # Criar gráfico de fallback simples
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers'))
            fig.update_layout(
                title="Erro ao gerar visualização",
                annotations=[
                    dict(
                        text=f"Ocorreu um erro: {str(e)}",
                        x=0.5,
                        y=0.5,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=14)
                    )
                ]
            )
            return fig
    
    def plot_backtest_results(self, backtest_results):
        """
        Plota os resultados do backtesting.
        
        Args:
            backtest_results: Dicionário com resultados do backtesting
            
        Returns:
            Figura plotly
        """
        try:
            # Verificar se temos dados completos
            if 'is_fallback' in backtest_results and backtest_results['is_fallback']:
                # Criar gráfico simples para fallback
                fig = go.Figure()
                
                fig.add_annotation(
                    text="Dados insuficientes para backtesting detalhado",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=14)
                )
                
                fig.update_layout(
                    title="Backtesting (Dados Limitados)",
                    height=400
                )
                
                return fig
            
            # Extrair dados
            dates = backtest_results['dates']
            predictions = backtest_results['predictions']
            actuals = backtest_results['actuals']
            
            # Criar figura
            fig = go.Figure()
            
            # Adicionar valores reais
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=actuals,
                    name="Preços Reais",
                    line=dict(color='blue', width=2)
                )
            )
            
            # Adicionar previsões
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=predictions,
                    name="Previsões do Modelo",
                    line=dict(color='red', width=2, dash='dot')
                )
            )
            
            # Layout
            fig.update_layout(
                title="Backtesting: Previsões vs. Valores Reais",
                xaxis_title="Data",
                yaxis_title="Preço (R$)",
                height=500,
                template="plotly_white",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro ao criar gráfico de backtesting: {str(e)}")
            
            # Criar gráfico de fallback simples
            fig = go.Figure()
            fig.add_annotation(
                text=f"Erro ao gerar visualização: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(
                title="Backtesting (Erro)",
                height=400
            )
            return fig
    
    def plot_comparison_chart(self, data_dict):
        """
        Plota gráfico de comparação entre diferentes ações.
        
        Args:
            data_dict: Dicionário com DataFrames de cada ação
            
        Returns:
            Figura plotly
        """
        try:
            # Criar figura
            fig = go.Figure()
            
            # Normalizar todos os preços para base 100
            for symbol, data in data_dict.items():
                if data is None or data.empty:
                    continue
                    
                # Pegar os últimos 90 dias
                recent_data = data.iloc[-90:].copy() if len(data) > 90 else data.copy()
                
                # Normalizar preços
                normalized = (recent_data['Close'] / recent_data['Close'].iloc[0]) * 100
                
                # Adicionar ao gráfico
                fig.add_trace(
                    go.Scatter(
                        x=recent_data.index,
                        y=normalized,
                        name=symbol,
                        line=dict(width=2)
                    )
                )
            
            # Layout
            fig.update_layout(
                title="Comparação de Performance (Base 100)",
                xaxis_title="Data",
                yaxis_title="Performance Relativa",
                height=500,
                template="plotly_white",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Erro ao criar gráfico de comparação: {str(e)}")
            
            # Criar gráfico de fallback simples
            fig = go.Figure()
            fig.add_annotation(
                text=f"Erro ao gerar visualização: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(
                title="Comparação de Ações (Erro)",
                height=400
            )
            return fig
