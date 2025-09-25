import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger('stock_prediction_app.utils')

class Utils:
    """
    Classe de utilitários para análises financeiras e processamento de dados.
    """
    
    def calculate_risk_metrics(self, data):
        """
        Calcula métricas de risco para a ação.
        
        Args:
            data: DataFrame com dados históricos
        
        Returns:
            Dicionário com métricas de risco
        """
        try:
            # Obter retornos diários
            returns = data['Returns'].dropna()
            
            if len(returns) < 30:
                logger.warning("Dados insuficientes para cálculo de métricas de risco")
                return self._fallback_risk_metrics()
            
            # Value at Risk (VaR) - 95%
            var_95 = np.percentile(returns, 5) * 100
            
            # Sharpe Ratio (anualizado)
            # Assumindo taxa livre de risco de 5% ao ano
            risk_free_rate = 0.05 / 252  # Taxa diária
            sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)
            
            # Maximum Drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / running_max - 1) * 100
            max_drawdown = drawdown.min()
            
            # Beta (em relação ao Ibovespa)
            # Nota: Simplificado pois não temos dados do Ibovespa
            # Aproximamos usando a volatilidade
            beta = returns.std() * 252 * 100
            
            return {
                'var_95': var_95,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volatility': returns.std() * np.sqrt(252) * 100,
                'beta': beta / 100  # Normalizado para escala típica
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular métricas de risco: {str(e)}")
            return self._fallback_risk_metrics()
    
    def _fallback_risk_metrics(self):
        """Métricas de risco padrão em caso de erro"""
        return {
            'var_95': -2.0,
            'sharpe_ratio': 0.5,
            'max_drawdown': -10.0,
            'volatility': 25.0,
            'beta': 1.0
        }
    
    def detect_significant_events(self, data):
        """
        Detecta eventos significativos no histórico da ação.
        
        Args:
            data: DataFrame com dados históricos
            
        Returns:
            Lista de eventos significativos
        """
        try:
            # Criar cópia e remover NaN
            df = data.dropna(subset=['Close', 'Returns']).copy()
            
            if len(df) < 30:
                return []
            
            events = []
            
            # 1. Grandes movimentos de preço (acima de 2 desvios padrão)
            returns_std = df['Returns'].std()
            threshold_up = returns_std * 2
            threshold_down = -returns_std * 2
            
            # Filtrar movimentos significativos
            significant_up = df[df['Returns'] > threshold_up]
            significant_down = df[df['Returns'] < threshold_down]
            
            # Adicionar eventos de alta
            for date, row in significant_up.iterrows():
                events.append({
                    'date': date,
                    'type': 'Alta Significativa',
                    'magnitude': row['Returns'] * 100,
                    'description': f"Alta de {row['Returns']*100:.2f}%, com volume {row['Volume_Ratio']:.1f}x acima da média."
                })
                
            # Adicionar eventos de baixa
            for date, row in significant_down.iterrows():
                events.append({
                    'date': date,
                    'type': 'Queda Significativa',
                    'magnitude': row['Returns'] * 100,
                    'description': f"Queda de {row['Returns']*100:.2f}%, com volume {row['Volume_Ratio']:.1f}x acima da média."
                })
                
            # 2. Cruzamentos relevantes de médias móveis
            # SMA 20 cruza acima do SMA 50
            cross_up = (df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1))
            
            # SMA 20 cruza abaixo do SMA 50
            cross_down = (df['SMA_20'] < df['SMA_50']) & (df['SMA_20'].shift(1) >= df['SMA_50'].shift(1))
            
            # Adicionar eventos de cruzamento
            for date, row in df[cross_up].iterrows():
                events.append({
                    'date': date,
                    'type': 'Cruzamento de Médias',
                    'magnitude': 0.0,
                    'description': "SMA 20 cruzou acima da SMA 50, sinal de tendência de alta."
                })
                
            for date, row in df[cross_down].iterrows():
                events.append({
                    'date': date,
                    'type': 'Cruzamento de Médias',
                    'magnitude': 0.0,
                    'description': "SMA 20 cruzou abaixo da SMA 50, sinal de tendência de baixa."
                })
                
            # 3. Eventos de RSI extremos
            # Sobrecomprado
            rsi_overbought = (df['RSI'] > 75) & (df['RSI'].shift(1) <= 75)
            
            # Sobrevendido
            rsi_oversold = (df['RSI'] < 25) & (df['RSI'].shift(1) >= 25)
            
            # Adicionar eventos de RSI
            for date, row in df[rsi_overbought].iterrows():
                events.append({
                    'date': date,
                    'type': 'RSI Sobrecomprado',
                    'magnitude': row['RSI'],
                    'description': f"RSI atingiu nível de sobrecompra ({row['RSI']:.1f}), possível reversão."
                })
                
            for date, row in df[rsi_oversold].iterrows():
                events.append({
                    'date': date,
                    'type': 'RSI Sobrevendido',
                    'magnitude': row['RSI'],
                    'description': f"RSI atingiu nível de sobrevenda ({row['RSI']:.1f}), possível reversão."
                })
                
            # 4. Volumes anormalmente altos
            high_volume = df[df['Volume_Ratio'] > 3]
            
            for date, row in high_volume.iterrows():
                events.append({
                    'date': date,
                    'type': 'Volume Anormal',
                    'magnitude': row['Volume_Ratio'],
                    'description': f"Volume {row['Volume_Ratio']:.1f}x acima da média, possível evento importante."
                })
                
            # Ordenar por data
            events = sorted(events, key=lambda x: x['date'], reverse=True)
            
            # Limitamos para apenas os 10 eventos mais recentes
            return events[:10]
            
        except Exception as e:
            logger.error(f"Erro ao detectar eventos significativos: {str(e)}")
            return []
        
    def filter_outliers(self, series, n_std=3):
        """
        Filtra outliers de uma série baseado em desvios padrão.
        
        Args:
            series: Série a ser filtrada
            n_std: Número de desvios padrão para considerar outlier
            
        Returns:
            Série filtrada
        """
        mean = series.mean()
        std = series.std()
        
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std
        
        return series[(series >= lower_bound) & (series <= upper_bound)]
    
    def format_large_number(self, num):
        """
        Formata grandes números para exibição (K, M, B).
        
        Args:
            num: Número a ser formatado
            
        Returns:
            String formatada
        """
        if num >= 1e9:
            return f"{num/1e9:.1f}B"
        elif num >= 1e6:
            return f"{num/1e6:.1f}M"
        elif num >= 1e3:
            return f"{num/1e3:.1f}K"
        else:
            return f"{num:.0f}"
