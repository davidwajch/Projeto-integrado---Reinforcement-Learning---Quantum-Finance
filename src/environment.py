"""
Ambiente de Reinforcement Learning para Trading
Define estados, ações e recompensas
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


class TradingEnvironment:
    """
    Ambiente de RL para trading de múltiplos ativos
    
    Estados:
        - Features técnicas normalizadas de cada ativo
        - Posições atuais (quantidade de cada ativo)
        - Capital disponível
    
    Ações:
        Para cada ativo: Comprar, Vender ou Manter (Hold)
        Total: 3^3 = 27 combinações possíveis (simplificado para 9 ações principais)
    
    Recompensas:
        - Lucro/prejuízo da operação
        - Penalização por risco excessivo
        - Bônus por diversificação
    """
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000.0,
                 transaction_cost: float = 0.001, max_position: float = 0.33,
                 original_prices: pd.DataFrame = None):
        """
        Inicializa o ambiente
        
        Args:
            data: DataFrame com features normalizadas
            initial_balance: Capital inicial
            transaction_cost: Custo de transação (%)
            max_position: Máximo de capital por ativo (%)
            original_prices: DataFrame com preços originais (não normalizados)
        """
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        
        # Extrai colunas de features (todas exceto Date)
        self.feature_columns = [col for col in data.columns if col != 'Date']
        
        # Extrai preços de fechamento originais (não normalizados)
        self.prices = {}
        if original_prices is not None:
            for ticker in ['VALE', 'PETR', 'BRFS']:
                price_col = f'{ticker}_Close'
                if price_col in original_prices.columns:
                    self.prices[ticker] = original_prices[price_col].values
        else:
            # Fallback: tenta usar dados normalizados (não ideal, mas funciona)
            # Assumindo que preços normalizados estão entre 0 e 1, precisamos desnormalizar
            # Por simplicidade, vamos usar um valor médio aproximado
            print("⚠️ Aviso: Preços originais não fornecidos. Usando valores aproximados.")
            for ticker in ['VALE', 'PETR', 'BRFS']:
                price_col = f'{ticker}_Close'
                if price_col in data.columns:
                    # Valores aproximados em R$ (será ajustado durante execução)
                    # VALE ~R$ 60, PETR ~R$ 30, BRFS ~R$ 15
                    base_prices = {'VALE': 60.0, 'PETR': 30.0, 'BRFS': 15.0}
                    normalized = data[price_col].values
                    self.prices[ticker] = normalized * base_prices.get(ticker, 50.0)
        
        # Define ações: [VALE_action, PETR_action, BRFS_action]
        # 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space_size = 27  # 3^3
        
        # Reset do ambiente
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reseta o ambiente para o estado inicial
        
        Returns:
            Estado inicial
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = {'VALE': 0, 'PETR': 0, 'BRFS': 0}  # Quantidade de ações
        self.portfolio_value = self.initial_balance
        self.total_profit = 0.0
        self.trades_history = []
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Retorna o estado atual
        
        Returns:
            Array numpy com features do estado atual
        """
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
        
        # Features técnicas do passo atual
        state_features = self.data.iloc[self.current_step][self.feature_columns].values
        
        # Adiciona informações de posição normalizadas
        position_features = []
        for ticker in ['VALE', 'PETR', 'BRFS']:
            if ticker in self.prices:
                current_price = self.prices[ticker][self.current_step]
                position_value = self.positions[ticker] * current_price
                position_ratio = position_value / self.portfolio_value if self.portfolio_value > 0 else 0
                position_features.append(position_ratio)
            else:
                position_features.append(0.0)
        
        # Capital disponível normalizado
        cash_ratio = self.balance / self.portfolio_value if self.portfolio_value > 0 else 1.0
        
        # Combina todas as features
        state = np.concatenate([state_features, position_features, [cash_ratio]])
        
        return state.astype(np.float32)
    
    def _decode_action(self, action: int) -> Tuple[int, int, int]:
        """
        Decodifica ação numérica em ações individuais para cada ativo
        
        Args:
            action: Ação codificada (0-26)
        
        Returns:
            Tupla (vale_action, petr_action, brfs_action)
        """
        # Converte ação para base 3
        vale_action = action % 3
        petr_action = (action // 3) % 3
        brfs_action = (action // 9) % 3
        
        return vale_action, petr_action, brfs_action
    
    def _execute_trade(self, ticker: str, action: int) -> float:
        """
        Executa uma operação de compra/venda
        
        Args:
            ticker: Nome do ticker
            action: 0=Hold, 1=Buy, 2=Sell
        
        Returns:
            Custo/receita da operação
        """
        if ticker not in self.prices:
            return 0.0
        
        current_price = self.prices[ticker][self.current_step]
        current_position = self.positions[ticker]
        
        if action == 0:  # Hold
            return 0.0
        
        elif action == 1:  # Buy
            # Calcula quanto pode comprar respeitando max_position
            max_investment = self.portfolio_value * self.max_position
            available_cash = self.balance
            
            investment = min(max_investment, available_cash * 0.95)  # Deixa 5% de reserva
            
            if investment > 0 and current_price > 0:
                shares_to_buy = int(investment / current_price)
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                
                if cost <= self.balance:
                    self.positions[ticker] += shares_to_buy
                    self.balance -= cost
                    return -cost
        
        elif action == 2:  # Sell
            if current_position > 0:
                shares_to_sell = int(current_position * 0.5)  # Vende 50% da posição
                if shares_to_sell > 0:
                    revenue = shares_to_sell * current_price * (1 - self.transaction_cost)
                    self.positions[ticker] -= shares_to_sell
                    self.balance += revenue
                    return revenue
        
        return 0.0
    
    def _calculate_portfolio_value(self) -> float:
        """
        Calcula o valor total do portfólio
        
        Returns:
            Valor total do portfólio
        """
        portfolio_value = self.balance
        
        for ticker in ['VALE', 'PETR', 'BRFS']:
            if ticker in self.prices and self.current_step < len(self.prices[ticker]):
                current_price = self.prices[ticker][self.current_step]
                portfolio_value += self.positions[ticker] * current_price
        
        return portfolio_value
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Executa um passo no ambiente
        
        Args:
            action: Ação a ser executada
        
        Returns:
            Tuple (next_state, reward, done, info)
        """
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0.0, True, {'message': 'Episode finished'}
        
        # Valor do portfólio antes da ação
        prev_portfolio_value = self._calculate_portfolio_value()
        
        # Decodifica e executa ações
        vale_action, petr_action, brfs_action = self._decode_action(action)
        
        # Executa trades
        vale_cost = self._execute_trade('VALE', vale_action)
        petr_cost = self._execute_trade('PETR', petr_action)
        brfs_cost = self._execute_trade('BRFS', brfs_action)
        
        # Avança para próximo passo
        self.current_step += 1
        
        # Calcula novo valor do portfólio
        self.portfolio_value = self._calculate_portfolio_value()
        
        # Calcula recompensa
        profit = self.portfolio_value - prev_portfolio_value
        reward = profit / self.initial_balance  # Normaliza pela capital inicial
        
        # Penaliza por manter muito capital parado
        cash_ratio = self.balance / self.portfolio_value if self.portfolio_value > 0 else 1.0
        if cash_ratio > 0.5:
            reward -= 0.001  # Pequena penalização
        
        # Bônus por diversificação
        active_positions = sum(1 for pos in self.positions.values() if pos > 0)
        if active_positions >= 2:
            reward += 0.0005  # Pequeno bônus
        
        # Verifica se episódio terminou
        done = self.current_step >= len(self.data) - 1
        
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'profit': profit,
            'step': self.current_step
        }
        
        return self._get_state(), reward, done, info
    
    def get_state_size(self) -> int:
        """
        Retorna o tamanho do vetor de estado
        
        Returns:
            Tamanho do estado
        """
        return len(self._get_state())
    
    def get_action_size(self) -> int:
        """
        Retorna o número de ações possíveis
        
        Returns:
            Número de ações
        """
        return self.action_space_size

