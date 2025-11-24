"""
Funções auxiliares para o projeto
"""

import numpy as np
import pandas as pd


def calculate_technical_indicators(df):
    """
    Calcula indicadores técnicos para análise
    
    Args:
        df: DataFrame com colunas ['Open', 'High', 'Low', 'Close', 'Volume']
    
    Returns:
        DataFrame com indicadores técnicos adicionados
    """
    df = df.copy()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'].fillna(50, inplace=True)
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD'].fillna(0, inplace=True)
    df['MACD_Signal'].fillna(0, inplace=True)
    
    # Médias móveis
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_20'].fillna(df['Close'], inplace=True)
    df['SMA_50'].fillna(df['Close'], inplace=True)
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Middle'].fillna(df['Close'], inplace=True)
    df['BB_Upper'].fillna(df['Close'], inplace=True)
    df['BB_Lower'].fillna(df['Close'], inplace=True)
    
    # Normalização de volume
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    df['Volume_Ratio'].fillna(1.0, inplace=True)
    
    # Retornos
    df['Returns'] = df['Close'].pct_change()
    df['Returns'].fillna(0, inplace=True)
    
    return df


def normalize_features(df, feature_columns):
    """
    Normaliza features usando min-max scaling
    
    Args:
        df: DataFrame com features
        feature_columns: Lista de colunas para normalizar
    
    Returns:
        DataFrame normalizado
    """
    df = df.copy()
    for col in feature_columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val - min_val > 0:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0.0
    return df


