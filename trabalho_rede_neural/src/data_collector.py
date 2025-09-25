import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
import time
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('stock_prediction_app.data_collector')

class DataCollector:
    """
    Classe responsável pela coleta de dados históricos das ações brasileiras.
    Implementa mecanismos de cache e fallback para lidar com problemas de conectividade.
    """
    
    def __init__(self):
        self.stocks = ["VIVA3.SA", "MOVI3.SA", "TRIS3.SA", "AMER3.SA", "MGLU3.SA"]
        self.stock_names = {
            "VIVA3.SA": "Vivara",
            "MOVI3.SA": "Movida",
            "TRIS3.SA": "Trisul",
            "AMER3.SA": "Americanas",
            "MGLU3.SA": "Magazine Luiza",
        }
        
        # Garantir que as pastas de dados existam
        self._ensure_data_directories()
    
    def _ensure_data_directories(self):
        """Cria as pastas necessárias para armazenar os dados das ações"""
        # Pasta principal de dados
        if not os.path.exists("data"):
            os.makedirs("data")
            
        # Pasta para dados de ações
        if not os.path.exists("data/stocks"):
            os.makedirs("data/stocks")
            
        # Pastas para cada ticker
        for stock in self.stocks:
            ticker_path = f"data/stocks/{stock.replace('.', '_')}"
            if not os.path.exists(ticker_path):
                os.makedirs(ticker_path)
    
    def get_stock_data(self, symbol, days=1825, offline_mode=False):
        """
        Coleta dados históricos de uma ação com mecanismo de cache.
        
        Args:
            symbol: Símbolo da ação
            days: Número de dias (histórico)
            offline_mode: Se True, usa apenas dados armazenados em cache
            
        Returns:
            DataFrame com dados históricos
        """
        ticker_dir = f"data/stocks/{symbol.replace('.', '_')}"
        cache_file = f"{ticker_dir}/historical_{days}d.parquet"
        
        # Verificar se temos dados em cache
        cached_data = self._check_cache(cache_file)
        
        # Se temos dados em cache e requisitado modo offline, ou se o cache é recente (< 24h)
        if cached_data is not None:
            cache_age = self._get_cache_age(cache_file)
            logger.info(f"Cache para {symbol} tem {cache_age:.1f} horas")
            
            # Se modo offline, ou se cache é recente (< 24h)
            if offline_mode or cache_age < 24:
                logger.info(f"Usando dados em cache para {symbol}")
                return cached_data
        
        # Se estamos no modo offline mas não temos cache, retorna None
        if offline_mode and cached_data is None:
            logger.warning(f"Modo offline ativado mas não há dados em cache para {symbol}")
            return None
        
        # Tentativa de buscar dados online
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            logger.info(f"Buscando dados online para {symbol} de {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
            
            # Adiciona timeout e retry para evitar bloqueios
            for attempt in range(3):  # Tenta até 3 vezes
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        interval='1d'
                    )
                    
                    if data.empty:
                        logger.warning(f"Nenhum dado encontrado para {symbol}")
                        # Se temos cache, use-o como fallback
                        if cached_data is not None:
                            logger.info(f"Usando cache como fallback para {symbol}")
                            return cached_data
                        return None
                    
                    # Sucesso! Limpar e preparar dados
                    data = self._clean_data(data)
                    data = self._add_derived_columns(data)
                    
                    # Salvar em cache
                    self._save_to_cache(data, cache_file)
                    logger.info(f"Dados atualizados para {symbol} e salvos em cache")
                    
                    # Também salva uma cópia CSV para fácil inspeção
                    csv_file = f"{ticker_dir}/historical_{days}d.csv"
                    data.to_csv(csv_file)
                    
                    return data
                    
                except Exception as e:
                    wait_time = (attempt + 1) * 2  # Backoff exponencial
                    logger.warning(f"Tentativa {attempt+1} falhou para {symbol}: {str(e)}. Aguardando {wait_time}s")
                    time.sleep(wait_time)
            
            # Se todas as tentativas falharam, use o cache como fallback
            if cached_data is not None:
                logger.warning(f"Todas as tentativas falharam para {symbol}. Usando cache antigo.")
                return cached_data
                
            logger.error(f"Falha ao obter dados para {symbol} após 3 tentativas")
            return None
                
        except Exception as e:
            logger.error(f"Erro ao coletar dados para {symbol}: {str(e)}")
            
            # Fallback para cache em caso de falha
            if cached_data is not None:
                logger.info(f"Usando cache como fallback para {symbol} após erro")
                return cached_data
                
            return None
    
    def _check_cache(self, cache_file):
        """Verifica se existe cache para a ação e carrega"""
        try:
            if os.path.exists(cache_file):
                # Carrega do arquivo parquet
                data = pd.read_parquet(cache_file)

                if data.empty:
                    logger.warning(f"Cache existente mas vazio: {cache_file}")
                    return None

                # Converte índice para datetime se necessário
                if not isinstance(data.index, pd.DatetimeIndex):
                    data.index = pd.to_datetime(data.index)

                return data

            # Fallback para CSV legado
            csv_file = cache_file.replace('.parquet', '.csv')
            if os.path.exists(csv_file):
                data = pd.read_csv(csv_file, index_col=0, parse_dates=True)

                if data.empty:
                    logger.warning(f"Cache CSV existente mas vazio: {csv_file}")
                    return None

                # Alguns arquivos CSV antigos podem não conter colunas derivadas
                if 'Returns' not in data.columns:
                    data = self._add_derived_columns(data)

                return data

            return None

        except Exception as e:
            logger.error(f"Erro ao carregar cache: {str(e)}")
            return None
    
    def _get_cache_age(self, cache_file):
        """Retorna a idade do cache em horas"""
        if not os.path.exists(cache_file):
            return float('inf')
            
        mod_time = os.path.getmtime(cache_file)
        current_time = time.time()
        
        # Idade em horas
        age_hours = (current_time - mod_time) / 3600
        return age_hours
    
    def _save_to_cache(self, data, cache_file):
        """Salva os dados em cache"""
        try:
            # Certifique-se de que o diretório existe
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            # Salvar como parquet (mais eficiente)
            data.to_parquet(cache_file)
            logger.info(f"Dados salvos em cache: {cache_file}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar cache: {str(e)}")
    
    def _clean_data(self, data):
        """Limpa e prepara os dados"""
        # Remover linhas com valores nulos
        data = data.dropna()
        
        # Remover dias sem volume
        data = data[data['Volume'] > 0]
        
        # Remover outliers extremos
        for col in ['Open', 'High', 'Low', 'Close']:
            Q1 = data[col].quantile(0.01)
            Q3 = data[col].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - (1.5 * IQR)
            upper_bound = Q3 + (1.5 * IQR)
            
            # Filtra valores dentro dos limites
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
        return data
    
    def _add_derived_columns(self, data):
        """Adiciona colunas derivadas aos dados"""
        # Retornos
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Volatilidade móvel
        data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # True Range para ATR
        data['TR1'] = data['High'] - data['Low']
        data['TR2'] = abs(data['High'] - data['Close'].shift(1))
        data['TR3'] = abs(data['Low'] - data['Close'].shift(1))
        data['True_Range'] = data[['TR1', 'TR2', 'TR3']].max(axis=1)
        data['ATR'] = data['True_Range'].rolling(window=14).mean()
        
        # Amplitude diária
        data['Daily_Range'] = ((data['High'] - data['Low']) / data['Close']) * 100
        
        # Gap de abertura
        data['Gap'] = ((data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)) * 100
        
        # Remover colunas temporárias
        data = data.drop(['TR1', 'TR2', 'TR3'], axis=1, errors='ignore')
        
        return data
    
    def get_multiple_stocks(self, days=1825, offline_mode=False):
        """Coleta dados para todas as ações."""
        data_dict = {}
        
        for symbol in self.stocks:
            logger.info(f"Coletando dados para {self.stock_names[symbol]} ({symbol})...")
            data = self.get_stock_data(symbol, days, offline_mode=offline_mode)
            
            if data is not None and not data.empty:
                data_dict[symbol] = data
                logger.info(f"✓ {len(data)} registros coletados para {symbol}")
            else:
                logger.warning(f"✗ Erro na coleta de {symbol}")
        
        return data_dict

    def download_all_stocks_data(self, days=1825, force_update=False):
        """
        Baixa e armazena dados para todas as ações.
        Útil para preparar o sistema para uso offline.
        
        Args:
            days: Número de dias (histórico)
            force_update: Se True, força atualização mesmo com cache recente
        """
        logger.info("Iniciando download de dados para todas as ações...")
        
        results = {}
        
        for symbol in self.stocks:
            try:
                ticker_dir = f"data/stocks/{symbol.replace('.', '_')}"
                cache_file = f"{ticker_dir}/historical_{days}d.parquet"
                
                # Verificar se temos dados em cache e se são recentes (< 24h)
                cache_age = float('inf')
                if os.path.exists(cache_file):
                    cache_age = self._get_cache_age(cache_file)
                
                # Se não forçar atualização e tiver cache recente, pula
                if not force_update and cache_age < 24:
                    logger.info(f"Cache recente para {symbol} (idade: {cache_age:.1f}h). Pulando...")
                    cached_data = self._check_cache(cache_file)
                    if cached_data is not None:
                        results[symbol] = {
                            "status": "cached",
                            "rows": len(cached_data),
                            "cache_age_hours": cache_age
                        }
                    continue
                
                # Baixa dados
                logger.info(f"Baixando dados para {symbol}...")
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d'
                )
                
                if data.empty:
                    logger.warning(f"Nenhum dado encontrado para {symbol}")
                    results[symbol] = {"status": "error", "message": "No data found"}
                    continue
                
                # Preparar dados
                data = self._clean_data(data)
                data = self._add_derived_columns(data)
                
                # Salvar em cache
                self._save_to_cache(data, cache_file)
                
                # Também salva uma cópia CSV para fácil inspeção
                csv_file = f"{ticker_dir}/historical_{days}d.csv"
                data.to_csv(csv_file)
                
                results[symbol] = {
                    "status": "downloaded",
                    "rows": len(data),
                    "start_date": data.index[0].strftime('%Y-%m-%d'),
                    "end_date": data.index[-1].strftime('%Y-%m-%d')
                }
                
                logger.info(f"✓ Dados para {symbol} atualizados e salvos ({len(data)} registros)")
                
                # Pausa entre requisições para não sobrecarregar a API
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Erro ao baixar dados para {symbol}: {str(e)}")
                results[symbol] = {"status": "error", "message": str(e)}
        
        return results
