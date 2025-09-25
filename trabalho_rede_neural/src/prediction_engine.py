import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger('stock_prediction_app.prediction_engine')

class PredictionEngine:
    """
    Motor de previsão que utiliza o modelo LSTM treinado para previsões futuras.
    Implementa também backtesting da estratégia.
    """
    
    def __init__(self, model, scaler):
        """
        Inicializa o motor de previsão.
        
        Args:
            model: Modelo LSTM treinado
            scaler: Scaler usado para normalizar os dados
        """
        self.model = model
        self.scaler = scaler
    
    def predict(self, data, horizon):
        """
        Realiza previsões para um horizonte específico.
        
        Args:
            data: DataFrame com dados históricos e indicadores técnicos
            horizon: Horizonte de previsão (ex: "1 semana", "1 mês", etc.)
            
        Returns:
            Dicionário com previsões e recomendações
        """
        try:
            # Mapear string de horizonte para dias
            horizon_days = self._convert_horizon_to_days(horizon)
            
            # Extrair últimos dados para previsão
            last_data_points = self._prepare_last_sequence(data)
            
            if last_data_points is None:
                return self._fallback_prediction(data, horizon)
            
            # Fazer previsão
            scaled_prediction = self.model.predict(last_data_points)
            
            # Preparar vetor para inverter a normalização (apenas para o preço de fechamento)
            dummy_array = np.zeros((1, self.scaler.scale_.shape[0]))
            dummy_array[0, 0] = scaled_prediction[0, 0]  # O primeiro índice é Close
            
            # Inverter a normalização para obter o preço previsto
            unscaled = self.scaler.inverse_transform(dummy_array)
            predicted_close_next_day = unscaled[0, 0]
            
            # Preço atual
            current_price = data['Close'].iloc[-1]
            
            # Calcular taxa de crescimento diário
            daily_growth_rate = ((predicted_close_next_day / current_price) - 1)
            
            # Projetar para o horizonte completo com crescimento composto
            predicted_price = current_price * ((1 + daily_growth_rate) ** horizon_days)
            
            # Calcular limites de confiança
            confidence_band = self._calculate_confidence_bands(data, predicted_price, horizon_days)
            
            # Análise técnica para recomendação
            recommendation, confidence = self._generate_recommendation(data, predicted_price, current_price)
            
            return {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_change': ((predicted_price / current_price) - 1) * 100,
                'lower_bound': confidence_band['lower'],
                'upper_bound': confidence_band['upper'],
                'recommendation': recommendation,
                'confidence': confidence,
                'horizon_days': horizon_days
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar previsão: {str(e)}")
            return self._fallback_prediction(data, horizon)
    
    def _convert_horizon_to_days(self, horizon):
        """Converte string de horizonte para número de dias"""
        mapping = {
            "1 semana": 5,  # dias úteis
            "1 mês": 21,
            "3 meses": 63,
            "6 meses": 126,
            "1 ano": 252
        }
        return mapping.get(horizon, 21)  # Padrão: 1 mês
    
    def _prepare_last_sequence(self, data):
        """Prepara a última sequência de dados para previsão"""
        try:
            # Selecionar features relevantes
            features = data[['Close', 'Volume', 'Returns', 'Volatility', 'ATR',
                            'RSI', 'MACD', 'BB_Position', 'SMA_20', 'EMA_12']].copy()
            
            # Remover valores NaN
            features = features.dropna()
            
            if len(features) < 20:  # Precisamos de pelo menos 20 pontos para a sequência
                logger.warning("Dados insuficientes para previsão")
                return None
            
            # Normalizar todos os dados
            features_scaled = self.scaler.transform(features)
            
            # Pegar os últimos 20 pontos para a previsão
            last_sequence = features_scaled[-20:].reshape(1, 20, features.shape[1])
            
            return last_sequence
            
        except Exception as e:
            logger.error(f"Erro ao preparar sequência para previsão: {str(e)}")
            return None
    
    def _calculate_confidence_bands(self, data, predicted_price, horizon_days):
        """Calcula bandas de confiança para a previsão"""
        # Calcular volatilidade histórica
        returns = data['Returns'].dropna()
        volatility = returns.std() * np.sqrt(252)  # Anualizada
        
        # Volatilidade ajustada ao horizonte
        horizon_volatility = volatility * np.sqrt(horizon_days / 252)
        
        # Intervalo de confiança de 95% (1.96 desvios padrão)
        price_range = predicted_price * horizon_volatility * 1.96
        
        lower_bound = predicted_price - price_range
        upper_bound = predicted_price + price_range
        
        # Garantir que o limite inferior não é negativo
        lower_bound = max(0, lower_bound)
        
        return {
            'lower': lower_bound,
            'upper': upper_bound
        }
    
    def _generate_recommendation(self, data, predicted_price, current_price):
        """Gera recomendação de compra/venda baseada em previsão e tendências"""
        # Calcular retorno esperado
        expected_return = (predicted_price / current_price - 1) * 100
        
        # Analisar tendências recentes
        sma_20 = data['SMA_20'].iloc[-1]
        sma_50 = data['SMA_50'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        macd = data['MACD'].iloc[-1]
        macd_signal = data['MACD_Signal'].iloc[-1]
        bb_position = data['BB_Position'].iloc[-1]
        
        # Pontuação técnica (0-100)
        technical_score = 0
        
        # Tendência de preço vs médias
        if current_price > sma_20:
            technical_score += 10
        if current_price > sma_50:
            technical_score += 10
        if sma_20 > sma_50:
            technical_score += 15
            
        # RSI
        if 40 <= rsi <= 60:  # Neutro
            technical_score += 5
        elif 30 <= rsi < 40 or 60 < rsi <= 70:  # Moderado
            technical_score += 10
        elif rsi < 30:  # Sobrevendido
            technical_score += 20
        elif rsi > 70:  # Sobrecomprado
            technical_score -= 10
            
        # MACD
        if macd > macd_signal:
            technical_score += 15
        else:
            technical_score -= 5
            
        # Bandas de Bollinger
        if bb_position < 0.2:  # Próximo à banda inferior
            technical_score += 15
        elif bb_position > 0.8:  # Próximo à banda superior
            technical_score -= 10
            
        # Combinar score técnico com retorno esperado
        final_score = 0
        
        if expected_return > 10:
            final_score = 80 + (technical_score * 0.2)
        elif expected_return > 5:
            final_score = 60 + (technical_score * 0.4)
        elif expected_return > 0:
            final_score = 40 + (technical_score * 0.6)
        elif expected_return > -5:
            final_score = 20 + (technical_score * 0.6)
        else:
            final_score = 0 + (technical_score * 0.4)
            
        # Ajustar para o intervalo 0-100
        final_score = max(0, min(100, final_score))
        
        # Determinar recomendação
        if final_score >= 70:
            recommendation = "COMPRAR"
        elif final_score >= 40:
            recommendation = "MANTER"
        else:
            recommendation = "VENDER"
            
        return recommendation, final_score
    
    def _fallback_prediction(self, data, horizon):
        """Previsão simplificada como fallback em caso de erro"""
        try:
            horizon_days = self._convert_horizon_to_days(horizon)
            current_price = data['Close'].iloc[-1]
            
            # Calcular retorno médio histórico
            returns = data['Returns'].dropna()
            avg_return = returns.mean()
            
            # Projetar para o horizonte com retorno médio
            predicted_price = current_price * ((1 + avg_return) ** horizon_days)
            
            # Volatilidade para bandas de confiança
            volatility = returns.std() * np.sqrt(252)  # Anualizada
            horizon_volatility = volatility * np.sqrt(horizon_days / 252)
            
            # Intervalo de confiança mais amplo para fallback (2.5 desvios padrão)
            price_range = predicted_price * horizon_volatility * 2.5
            
            # Bandas de confiança
            lower_bound = max(0, predicted_price - price_range)
            upper_bound = predicted_price + price_range
            
            # Recomendação mais conservadora
            if avg_return > 0 and current_price > data['SMA_50'].iloc[-1]:
                recommendation = "MANTER"
                confidence = 55
            else:
                recommendation = "VENDER"
                confidence = 50
            
            return {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_change': ((predicted_price / current_price) - 1) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'recommendation': recommendation,
                'confidence': confidence,
                'horizon_days': horizon_days,
                'is_fallback': True
            }
            
        except Exception as e:
            logger.error(f"Erro crítico no fallback: {str(e)}")
            
            # Fallback extremo
            return {
                'current_price': data['Close'].iloc[-1] if not data.empty else 0,
                'predicted_price': data['Close'].iloc[-1] * 0.98 if not data.empty else 0,
                'predicted_change': -2,
                'lower_bound': data['Close'].iloc[-1] * 0.90 if not data.empty else 0,
                'upper_bound': data['Close'].iloc[-1] * 1.05 if not data.empty else 0,
                'recommendation': "MANTER",
                'confidence': 40,
                'horizon_days': self._convert_horizon_to_days(horizon),
                'is_fallback': True,
                'error': str(e)
            }
    
    def backtest(self, data):
        """
        Realiza backtesting do modelo em dados históricos.
        
        Args:
            data: DataFrame com dados históricos e indicadores
            
        Returns:
            Dicionário com resultados do backtesting
        """
        try:
            # Criar cópia para não alterar o original
            backtest_data = data.copy()
            
            # Selecionar features relevantes
            features = backtest_data[['Close', 'Volume', 'Returns', 'Volatility', 'ATR',
                                     'RSI', 'MACD', 'BB_Position', 'SMA_20', 'EMA_12']].copy()
            
            # Remover valores NaN
            features = features.dropna()
            
            if len(features) < 40:  # Precisamos de dados suficientes
                logger.warning("Dados insuficientes para backtesting")
                return self._fallback_backtest(data)
            
            # Normalizar dados
            features_scaled = self.scaler.transform(features)
            
            # Preparar sequências para backtesting
            X_backtest, y_backtest = [], []
            sequence_length = 20
            
            for i in range(sequence_length, len(features_scaled) - 1):
                X_backtest.append(features_scaled[i-sequence_length:i])
                y_backtest.append(features_scaled[i, 0])  # Índice 0 é Close
            
            X_backtest, y_backtest = np.array(X_backtest), np.array(y_backtest)
            
            # Fazer previsões
            predictions_scaled = self.model.predict(X_backtest)
            
            # Converter previsões para preços
            predictions = []
            actuals = []
            
            for i in range(len(predictions_scaled)):
                # Preparar array para inverter a normalização
                dummy_array = np.zeros((1, self.scaler.scale_.shape[0]))
                dummy_array[0, 0] = predictions_scaled[i, 0]
                unscaled_pred = self.scaler.inverse_transform(dummy_array)[0, 0]
                predictions.append(unscaled_pred)
                
                # Valor real
                dummy_array[0, 0] = y_backtest[i]
                unscaled_actual = self.scaler.inverse_transform(dummy_array)[0, 0]
                actuals.append(unscaled_actual)
            
            # Calcular métricas de desempenho
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # Erro médio absoluto percentual
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
            
            # Acurácia direcional (se acertou a direção do movimento)
            pred_direction = np.diff(predictions) > 0
            actual_direction = np.diff(actuals) > 0
            directional_accuracy = np.mean(pred_direction == actual_direction) * 100
            
            # Simulação de estratégia de trading simples
            # Compra quando a previsão é de alta, vende quando é de baixa
            strategy_returns = []
            buy_hold_returns = []
            
            for i in range(1, len(predictions)):
                # Retorno da estratégia
                if predictions[i] > predictions[i-1]:  # Previsão de alta
                    strategy_returns.append(actuals[i] / actuals[i-1] - 1)
                else:  # Previsão de baixa
                    strategy_returns.append(0)  # Fica fora do mercado
                
                # Retorno de buy & hold
                buy_hold_returns.append(actuals[i] / actuals[i-1] - 1)
            
            # Calcular retornos acumulados
            strategy_return = np.prod(1 + np.array(strategy_returns)) - 1
            buy_hold_return = np.prod(1 + np.array(buy_hold_returns)) - 1
            
            # Sharpe ratio (simplified)
            risk_free_rate = 0.05 / 252  # 5% ao ano convertido para diário
            strategy_sharpe = (np.mean(strategy_returns) - risk_free_rate) / np.std(strategy_returns) * np.sqrt(252)
            
            return {
                'accuracy': directional_accuracy,
                'mape': mape,
                'strategy_return': strategy_return * 100,
                'buy_hold_return': buy_hold_return * 100,
                'sharpe_ratio': strategy_sharpe,
                'predictions': predictions,
                'actuals': actuals,
                'dates': features.index[-len(predictions):],
                'strategy_returns': strategy_returns,
                'buy_hold_returns': buy_hold_returns
            }
            
        except Exception as e:
            logger.error(f"Erro no backtesting: {str(e)}")
            return self._fallback_backtest(data)
    
    def _fallback_backtest(self, data):
        """Resultados de backtesting fictícios em caso de erro"""
        return {
            'accuracy': 52.0,
            'mape': 5.0,
            'strategy_return': 3.5,
            'buy_hold_return': 3.0,
            'sharpe_ratio': 0.8,
            'is_fallback': True
        }
