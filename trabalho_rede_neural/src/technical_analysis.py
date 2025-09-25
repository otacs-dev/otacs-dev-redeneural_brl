import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('stock_prediction_app.technical_analysis')

class TechnicalAnalysis:
    """
    Classe para cálculo de indicadores técnicos para análise de ações.
    """
    
    def __init__(self):
        pass
    
    def calculate_indicators(self, data):
        """
        Calcula todos os indicadores técnicos para o DataFrame fornecido.
        
        Args:
            data: DataFrame com dados históricos
            
        Returns:
            DataFrame com indicadores técnicos adicionados
        """
        df = data.copy()
        
        try:
            # Médias móveis
            df = self._add_moving_averages(df)
            
            # Osciladores
            df = self._add_oscillators(df)
            
            # Bandas de Bollinger
            df = self._add_bollinger_bands(df)
            
            # MACD
            df = self._add_macd(df)
            
            # Indicadores de volume
            df = self._add_volume_indicators(df)
            
            # Suporte e resistência
            df = self._add_support_resistance(df)
            
            # Indicadores customizados
            df = self._add_custom_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Erro no cálculo de indicadores técnicos: {str(e)}")
            return data
    
    def _add_moving_averages(self, df):
        """Adiciona médias móveis"""
        # Médias móveis simples
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Médias móveis exponenciais
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        
        # Posição relativa às médias
        df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20'] * 100
        df['Price_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50'] * 100
        
        return df
    
    def _add_oscillators(self, df):
        """Adiciona osciladores"""
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_Oversold'] = df['RSI'] < 30
        df['RSI_Overbought'] = df['RSI'] > 70
        
        # Stochastic %K
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Williams %R
        df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
        
        return df
    
    def _add_bollinger_bands(self, df):
        """Adiciona Bandas de Bollinger"""
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Posição dentro das bandas
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Largura das bandas
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        return df
    
    def _add_macd(self, df):
        """Adiciona MACD"""
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Sinais de cruzamento
        df['MACD_Bullish'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
        df['MACD_Bearish'] = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
        
        return df
    
    def _add_volume_indicators(self, df):
        """Adiciona indicadores de volume"""
        # Volume médio
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # On Balance Volume (simplificado)
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        df['OBV'] = obv
        df['OBV_SMA'] = pd.Series(obv).rolling(window=20).mean().values
        
        return df
    
    def _add_support_resistance(self, df):
        """Adiciona níveis de suporte e resistência"""
        # Identificar máximos e mínimos locais (simplificado)
        window = 10
        
        df['Local_Max'] = df['High'].rolling(window=2*window+1, center=True).max()
        df['Local_Min'] = df['Low'].rolling(window=2*window+1, center=True).min()
        
        # Flag para máximos e mínimos locais
        df['Is_Local_Max'] = (df['High'] == df['Local_Max']) & (df['High'] != df['High'].shift(1))
        df['Is_Local_Min'] = (df['Low'] == df['Local_Min']) & (df['Low'] != df['Low'].shift(1))
        
        # Cálculo simplificado de suporte e resistência
        # (usando médias das mínimas e máximas recentes)
        df['Resistance'] = df['High'].rolling(window=50).max()
        df['Support'] = df['Low'].rolling(window=50).min()
        
        return df
    
    def _add_custom_indicators(self, df):
        """Adiciona indicadores customizados"""
        # Indicador de tendência: Combinação de médias móveis
        df['Trend_Indicator'] = (
            (df['SMA_10'] > df['SMA_20']).astype(int) + 
            (df['SMA_20'] > df['SMA_50']).astype(int) + 
            (df['SMA_50'] > df['SMA_200']).astype(int)
        ) / 3
        
        # Flag de reversão
        df['Reversal_Down'] = (df['Trend_Indicator'] >= 0.66) & (df['RSI'] > 70)
        df['Reversal_Up'] = (df['Trend_Indicator'] <= 0.33) & (df['RSI'] < 30)
        
        # Médias móveis de Momentum 
        df['Momentum'] = df['Close'] / df['Close'].shift(10) - 1
        df['Momentum_SMA'] = df['Momentum'].rolling(window=10).mean()
        
        # Índice de Força: Preço x Volume
        df['Force_Index'] = df['Close'].diff() * df['Volume']
        df['Force_Index_EMA'] = df['Force_Index'].ewm(span=13).mean()
        
        return df
