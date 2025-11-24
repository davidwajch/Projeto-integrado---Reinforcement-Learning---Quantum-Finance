"""
Módulo para download e processamento de dados históricos das ações
"""

import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta
from src.utils import calculate_technical_indicators, normalize_features


class DataLoader:
    """
    Classe para carregar e processar dados históricos das ações
    """
    
    # Tickers das ações na B3
    TICKERS = {
        'VALE': 'VALE3.SA',
        'PETR': 'PETR4.SA',
        'BRFS': 'BRFS3.SA'
    }
    
    def __init__(self, data_dir='data'):
        """
        Inicializa o DataLoader
        
        Args:
            data_dir: Diretório para salvar/carregar dados
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_data(self, ticker_name, start_date=None, end_date=None, period='2y'):
        """
        Baixa dados históricos de uma ação
        
        Args:
            ticker_name: Nome do ticker ('VALE', 'PETR', 'BRFS')
            start_date: Data inicial (YYYY-MM-DD)
            end_date: Data final (YYYY-MM-DD)
            period: Período padrão se não especificar datas ('2y', '1y', etc.)
        
        Returns:
            DataFrame com dados históricos
        """
        ticker_symbol = self.TICKERS.get(ticker_name.upper())
        if not ticker_symbol:
            raise ValueError(f"Ticker {ticker_name} não encontrado")
        
        # Define período padrão se não especificado
        if start_date is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)  # 2 anos
        
        try:
            ticker = yf.Ticker(ticker_symbol)
            if start_date and end_date:
                df = ticker.history(start=start_date, end=end_date)
            else:
                df = ticker.history(period=period)
            
            if df.empty:
                raise ValueError(f"Nenhum dado encontrado para {ticker_name}")
            
            # Renomeia colunas para padrão
            df.columns = [col.replace(' ', '') for col in df.columns]
            df.reset_index(inplace=True)
            
            # Salva dados
            file_path = os.path.join(self.data_dir, f"{ticker_name.lower()}_data.csv")
            df.to_csv(file_path, index=False)
            
            print(f"Dados de {ticker_name} baixados: {len(df)} registros")
            return df
            
        except Exception as e:
            print(f"Erro ao baixar dados de {ticker_name}: {e}")
            raise
    
    def load_all_data(self, start_date=None, end_date=None, period='2y'):
        """
        Baixa dados de todas as ações
        
        Args:
            start_date: Data inicial
            end_date: Data final
            period: Período padrão
        
        Returns:
            Dicionário com DataFrames de cada ação
        """
        data = {}
        for ticker_name in self.TICKERS.keys():
            try:
                df = self.download_data(ticker_name, start_date, end_date, period)
                # Calcula indicadores técnicos
                df = calculate_technical_indicators(df)
                data[ticker_name] = df
            except Exception as e:
                print(f"Erro ao processar {ticker_name}: {e}")
        
        return data
    
    def prepare_features(self, data_dict):
        """
        Prepara features normalizadas para o ambiente de RL
        
        Args:
            data_dict: Dicionário com DataFrames de cada ação
        
        Returns:
            Tupla (DataFrame com features normalizadas, DataFrame com preços originais)
        """
        feature_columns = [
            'Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal',
            'SMA_20', 'SMA_50', 'BB_Upper', 'BB_Lower', 'BB_Middle',
            'Volume_Ratio', 'Returns'
        ]
        
        # Combina dados de todas as ações
        combined_data = []
        prices_dict = {}
        
        for ticker_name, df in data_dict.items():
            df_ticker = df.copy()
            
            # Salva preços originais antes de normalizar
            if 'Close' in df_ticker.columns:
                prices_dict[ticker_name] = df_ticker['Close'].values
            
            # Adiciona prefixo ao nome das colunas
            for col in feature_columns:
                if col in df_ticker.columns:
                    df_ticker[f'{ticker_name}_{col}'] = df_ticker[col]
            
            # Mantém apenas colunas com prefixo
            ticker_cols = [col for col in df_ticker.columns if ticker_name in col or col == 'Date']
            df_ticker = df_ticker[ticker_cols]
            combined_data.append(df_ticker)
        
        # Merge de todos os DataFrames
        result = combined_data[0]
        for df in combined_data[1:]:
            result = pd.merge(result, df, on='Date', how='outer')
        
        # Ordena por data
        result.sort_values('Date', inplace=True)
        result.reset_index(drop=True, inplace=True)
        
        # Preenche valores faltantes
        result.ffill(inplace=True)
        result.bfill(inplace=True)
        
        # Cria DataFrame com preços originais alinhados
        # Usa as datas do resultado final e faz merge com preços originais
        prices_df = pd.DataFrame({'Date': result['Date']})
        
        for ticker_name, df_original in data_dict.items():
            if 'Close' in df_original.columns:
                # Cria DataFrame temporário com preços originais
                temp_df = pd.DataFrame({
                    'Date': df_original['Date'],
                    f'{ticker_name}_Close': df_original['Close']
                })
                # Faz merge com as datas do resultado
                prices_df = pd.merge(prices_df, temp_df, on='Date', how='left')
        
        # Preenche valores faltantes
        prices_df.ffill(inplace=True)
        prices_df.bfill(inplace=True)
        
        # Normaliza features
        all_feature_cols = [col for col in result.columns if col != 'Date']
        result = normalize_features(result, all_feature_cols)
        
        return result, prices_df
    
    def load_from_file(self, ticker_name):
        """
        Carrega dados de arquivo CSV local
        
        Args:
            ticker_name: Nome do ticker
        
        Returns:
            DataFrame com dados
        """
        file_path = os.path.join(self.data_dir, f"{ticker_name.lower()}_data.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        else:
            raise FileNotFoundError(f"Arquivo {file_path} não encontrado")

