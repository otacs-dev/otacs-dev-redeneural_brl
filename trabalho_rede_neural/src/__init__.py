"""
Sistema de Previsão de Ações Brasileiras
----------------------------------------

Este pacote contém módulos para análise e previsão de ações brasileiras,
incluindo coleta de dados, análise técnica, modelagem LSTM e visualizações.
"""

from src.data_collector import DataCollector
from src.lstm_model import LSTMModel
from src.technical_analysis import TechnicalAnalysis
from src.prediction_engine import PredictionEngine
from src.visualizations import Visualizations
from src.utils import Utils

__version__ = "1.0.0"
