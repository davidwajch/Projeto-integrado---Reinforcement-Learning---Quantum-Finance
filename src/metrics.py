"""
Métricas de avaliação financeira
"""

import numpy as np
import pandas as pd
from typing import List, Dict


class FinancialMetrics:
    """
    Classe para calcular métricas financeiras de desempenho
    """
    
    @staticmethod
    def calculate_returns(portfolio_values: List[float]) -> np.ndarray:
        """
        Calcula retornos do portfólio
        
        Args:
            portfolio_values: Lista de valores do portfólio ao longo do tempo
        
        Returns:
            Array com retornos percentuais
        """
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        return returns
    
    @staticmethod
    def total_return(initial_value: float, final_value: float) -> float:
        """
        Calcula retorno total
        
        Args:
            initial_value: Valor inicial
            final_value: Valor final
        
        Returns:
            Retorno total percentual
        """
        return ((final_value - initial_value) / initial_value) * 100
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Calcula Sharpe Ratio
        
        Args:
            returns: Array de retornos
            risk_free_rate: Taxa livre de risco (anual)
        
        Returns:
            Sharpe Ratio anualizado
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        # Retorno médio anualizado (assumindo 252 dias úteis)
        mean_return = np.mean(returns) * 252
        
        # Desvio padrão anualizado
        std_return = np.std(returns) * np.sqrt(252)
        
        if std_return == 0:
            return 0.0
        
        sharpe = (mean_return - risk_free_rate) / std_return
        return sharpe
    
    @staticmethod
    def maximum_drawdown(portfolio_values: List[float]) -> float:
        """
        Calcula Maximum Drawdown
        
        Args:
            portfolio_values: Lista de valores do portfólio
        
        Returns:
            Maximum Drawdown percentual
        """
        portfolio_values = np.array(portfolio_values)
        
        # Calcula running maximum
        running_max = np.maximum.accumulate(portfolio_values)
        
        # Calcula drawdown
        drawdown = (portfolio_values - running_max) / running_max
        
        # Maximum drawdown
        max_dd = np.min(drawdown) * 100
        
        return abs(max_dd)
    
    @staticmethod
    def win_rate(returns: np.ndarray) -> float:
        """
        Calcula taxa de acerto (win rate)
        
        Args:
            returns: Array de retornos
        
        Returns:
            Win rate percentual
        """
        if len(returns) == 0:
            return 0.0
        
        wins = np.sum(returns > 0)
        total = len(returns)
        
        return (wins / total) * 100
    
    @staticmethod
    def volatility(returns: np.ndarray) -> float:
        """
        Calcula volatilidade anualizada
        
        Args:
            returns: Array de retornos
        
        Returns:
            Volatilidade percentual anualizada
        """
        if len(returns) == 0:
            return 0.0
        
        return np.std(returns) * np.sqrt(252) * 100
    
    @staticmethod
    def calculate_all_metrics(portfolio_values: List[float],
                             initial_value: float = None,
                             risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Calcula todas as métricas financeiras
        
        Args:
            portfolio_values: Lista de valores do portfólio
            initial_value: Valor inicial (se None, usa primeiro valor)
            risk_free_rate: Taxa livre de risco
        
        Returns:
            Dicionário com todas as métricas
        """
        if initial_value is None:
            initial_value = portfolio_values[0] if portfolio_values else 0
        
        final_value = portfolio_values[-1] if portfolio_values else initial_value
        
        returns = FinancialMetrics.calculate_returns(portfolio_values)
        
        metrics = {
            'Total Return (%)': FinancialMetrics.total_return(initial_value, final_value),
            'Sharpe Ratio': FinancialMetrics.sharpe_ratio(returns, risk_free_rate),
            'Maximum Drawdown (%)': FinancialMetrics.maximum_drawdown(portfolio_values),
            'Win Rate (%)': FinancialMetrics.win_rate(returns),
            'Volatility (%)': FinancialMetrics.volatility(returns),
            'Final Value': final_value,
            'Initial Value': initial_value
        }
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float]):
        """
        Imprime métricas formatadas
        
        Args:
            metrics: Dicionário com métricas
        """
        print("\n" + "="*50)
        print("MÉTRICAS DE DESEMPENHO")
        print("="*50)
        for key, value in metrics.items():
            if 'Ratio' in key:
                print(f"{key}: {value:.4f}")
            elif 'Value' in key:
                print(f"{key}: R$ {value:,.2f}")
            else:
                print(f"{key}: {value:.2f}")
        print("="*50 + "\n")


