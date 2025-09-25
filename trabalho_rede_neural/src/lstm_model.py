import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import os
import logging
import pickle

logger = logging.getLogger('stock_prediction_app.lstm_model')

class LSTMModel:
    """
    Classe para criar, treinar e avaliar modelos LSTM para previsão de preços de ações.
    """
    
    def __init__(self):
        # Garantir reprodutibilidade
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Diretório para salvar modelos
        self.models_dir = "data/models"
        
        # Criar diretório se não existir
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def train(self, data, stock_symbol, force_retrain=False):
        """
        Treina ou carrega um modelo LSTM para previsão de preços.
        
        Args:
            data: DataFrame com dados históricos e indicadores técnicos
            stock_symbol: Símbolo da ação
            force_retrain: Se True, força o retreinamento mesmo havendo modelo salvo
            
        Returns:
            Dicionário com o modelo e o scaler para normalização
        """
        # Verificar se já existe um modelo salvo
        model_path = f"{self.models_dir}/{stock_symbol.replace('.', '_')}_lstm_model"
        
        if os.path.exists(f"{model_path}.h5") and not force_retrain:
            return self._load_model(stock_symbol)
        
        logger.info(f"Treinando novo modelo LSTM para {stock_symbol}...")
        
        try:
            # Preparar dados para treinamento
            X, y, scaler = self._prepare_data(data)
            
            if X is None or y is None:
                logger.error(f"Falha ao preparar dados para {stock_symbol}")
                return None
            
            # Dividir em conjuntos de treinamento e teste
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Criar e treinar modelo
            model = self._create_model(X.shape[1], X.shape[2])
            
            # Early stopping para evitar overfitting
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            )
            
            # Reduzir learning rate quando o treinamento estagnar
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            )
            
            # Treinar modelo
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,
                batch_size=32,
                verbose=0,
                callbacks=[early_stopping, reduce_lr]
            )
            
            # Avaliar modelo
            loss = model.evaluate(X_test, y_test, verbose=0)
            logger.info(f"Modelo treinado para {stock_symbol}. Loss: {loss}")
            
            # Salvar modelo e scaler
            self._save_model(model, scaler, stock_symbol)
            
            # Retornar para uso
            return {"model": model, "scaler": scaler}
            
        except Exception as e:
            logger.error(f"Erro ao treinar modelo LSTM para {stock_symbol}: {str(e)}")
            return None
    
    def _prepare_data(self, data):
        """
        Prepara os dados para o modelo LSTM.
        """
        try:
            # Selecionar features a serem utilizadas
            features = data[['Close', 'Volume', 'Returns', 'Volatility', 'ATR',
                            'RSI', 'MACD', 'BB_Position', 'SMA_20', 'EMA_12']].copy()
            
            # Remover valores NaN
            features = features.dropna()
            
            if len(features) < 100:
                logger.warning("Dados insuficientes para treinamento do modelo LSTM")
                return None, None, None
            
            # Normalizar dados
            scaler = MinMaxScaler(feature_range=(0, 1))
            features_scaled = scaler.fit_transform(features)
            
            # Preparar sequências de treinamento
            X, y = [], []
            sequence_length = 20  # 20 dias de dados para prever o próximo
            
            for i in range(sequence_length, len(features_scaled)):
                X.append(features_scaled[i-sequence_length:i])
                y.append(features_scaled[i, 0])  # Índice 0 corresponde a 'Close'
            
            X, y = np.array(X), np.array(y)
            
            return X, y, scaler
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados: {str(e)}")
            return None, None, None
    
    def _create_model(self, n_timesteps, n_features):
        """
        Cria a arquitetura do modelo LSTM.
        """
        model = Sequential([
            # Primeira camada LSTM
            LSTM(50, return_sequences=True, 
                 input_shape=(n_timesteps, n_features)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Segunda camada LSTM
            LSTM(50, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            # Camadas densas de saída
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        # Compilar modelo
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error'
        )
        
        return model
    
    def _save_model(self, model, scaler, stock_symbol):
        """
        Salva o modelo treinado e o scaler para uso futuro.
        """
        try:
            # Criar diretório se não existir
            if not os.path.exists(self.models_dir):
                os.makedirs(self.models_dir)
            
            # Salvar modelo
            model_file = f"{self.models_dir}/{stock_symbol.replace('.', '_')}_lstm_model.h5"
            model.save(model_file)
            
            # Salvar scaler
            scaler_file = f"{self.models_dir}/{stock_symbol.replace('.', '_')}_scaler.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            
            logger.info(f"Modelo e scaler salvos para {stock_symbol}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {str(e)}")
    
    def _load_model(self, stock_symbol):
        """
        Carrega um modelo salvo anteriormente.
        """
        try:
            model_file = f"{self.models_dir}/{stock_symbol.replace('.', '_')}_lstm_model.h5"
            scaler_file = f"{self.models_dir}/{stock_symbol.replace('.', '_')}_scaler.pkl"
            
            # Verificar se os arquivos existem
            if not os.path.exists(model_file) or not os.path.exists(scaler_file):
                logger.warning(f"Arquivos de modelo não encontrados para {stock_symbol}")
                return None
            
            # Carregar modelo
            model = tf.keras.models.load_model(model_file)
            
            # Carregar scaler
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            
            logger.info(f"Modelo carregado para {stock_symbol}")
            
            return {"model": model, "scaler": scaler}
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            return None
