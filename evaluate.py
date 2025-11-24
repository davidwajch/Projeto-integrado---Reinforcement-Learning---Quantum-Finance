"""
Script de avaliação do agente DQN treinado
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.data_loader import DataLoader
from src.environment import TradingEnvironment
from src.agent import DQNAgent
from src.metrics import FinancialMetrics

# Configurações
INITIAL_BALANCE = 100000.0
MODEL_DIR = 'models'
DATA_DIR = 'data'
REPORTS_DIR = 'reports'

def evaluate_agent():
    """
    Função principal de avaliação
    """
    print("="*60)
    print("AVALIAÇÃO DO AGENTE DQN")
    print("="*60)
    
    # Verifica se modelo existe
    model_path = os.path.join(MODEL_DIR, 'dqn_model.pth')
    if not os.path.exists(model_path):
        print(f"\n❌ Erro: Modelo não encontrado em {model_path}")
        print("   Execute train.py primeiro para treinar o agente.")
        return
    
    # Carrega dados
    print("\n1. Carregando dados históricos...")
    data_loader = DataLoader(data_dir=DATA_DIR)
    
    try:
        data_dict = {}
        for ticker in ['VALE', 'PETR', 'BRFS']:
            try:
                df = data_loader.load_from_file(ticker)
                from src.utils import calculate_technical_indicators
                df = calculate_technical_indicators(df)
                data_dict[ticker] = df
                print(f"   ✓ Dados de {ticker} carregados")
            except FileNotFoundError:
                print(f"   ⚠ Baixando dados de {ticker}...")
                df = data_loader.download_data(ticker, period='2y')
                from src.utils import calculate_technical_indicators
                df = calculate_technical_indicators(df)
                data_dict[ticker] = df
    except Exception as e:
        print(f"   ⚠ Erro: {e}")
        return
    
    # Prepara features
    print("\n2. Preparando features...")
    features_df, prices_df = data_loader.prepare_features(data_dict)
    print(f"   ✓ Features preparadas: {len(features_df)} registros")
    
    # Cria ambiente
    print("\n3. Criando ambiente de RL...")
    env = TradingEnvironment(features_df, initial_balance=INITIAL_BALANCE, original_prices=prices_df)
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    
    # Carrega agente
    print("\n4. Carregando agente treinado...")
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size
    )
    agent.load(model_path)
    agent.epsilon = 0.0  # Desativa exploração para avaliação
    print(f"   ✓ Agente carregado")
    
    # Avaliação
    print("\n5. Executando avaliação...")
    print("-"*60)
    
    state = env.reset()
    done = False
    portfolio_values = [env.portfolio_value]
    actions_taken = []
    trades_history = []
    
    step = 0
    while not done:
        action = agent.act(state, training=False)
        next_state, reward, done, info = env.step(action)
        
        portfolio_values.append(info['portfolio_value'])
        actions_taken.append(action)
        
        # Registra trades
        if info.get('positions'):
            trades_history.append({
                'step': step,
                'portfolio_value': info['portfolio_value'],
                'balance': info['balance'],
                'positions': info['positions'].copy(),
                'profit': info.get('profit', 0)
            })
        
        state = next_state
        step += 1
        
        if step % 50 == 0:
            print(f"   Step {step}/{len(features_df)} | "
                  f"Portfólio: R$ {info['portfolio_value']:,.2f} | "
                  f"Retorno: {FinancialMetrics.total_return(INITIAL_BALANCE, info['portfolio_value']):.2f}%")
    
    print("-"*60)
    
    # Calcula métricas
    print("\n6. Calculando métricas...")
    metrics = FinancialMetrics.calculate_all_metrics(
        portfolio_values,
        initial_value=INITIAL_BALANCE
    )
    FinancialMetrics.print_metrics(metrics)
    
    # Gera visualizações
    print("\n7. Gerando visualizações...")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    plot_evaluation_results(portfolio_values, features_df, metrics)
    
    # Salva resultados
    save_results(portfolio_values, trades_history, metrics, features_df)
    
    print("\n✓ Avaliação concluída com sucesso!")
    print(f"✓ Relatórios salvos em: {REPORTS_DIR}/")


def plot_evaluation_results(portfolio_values, features_df, metrics):
    """
    Gera gráficos de avaliação
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Evolução do portfólio
    axes[0, 0].plot(portfolio_values, linewidth=2, label='Portfólio')
    axes[0, 0].axhline(y=INITIAL_BALANCE, color='r', linestyle='--', label='Capital Inicial')
    axes[0, 0].set_title('Evolução do Valor do Portfólio')
    axes[0, 0].set_xlabel('Dias')
    axes[0, 0].set_ylabel('Valor (R$)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Retornos diários
    returns = FinancialMetrics.calculate_returns(portfolio_values)
    axes[0, 1].plot(returns * 100, alpha=0.7)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_title('Retornos Diários')
    axes[0, 1].set_xlabel('Dias')
    axes[0, 1].set_ylabel('Retorno (%)')
    axes[0, 1].grid(True)
    
    # Distribuição de retornos
    axes[1, 0].hist(returns * 100, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].set_title('Distribuição de Retornos')
    axes[1, 0].set_xlabel('Retorno (%)')
    axes[1, 0].set_ylabel('Frequência')
    axes[1, 0].grid(True)
    
    # Drawdown
    portfolio_array = np.array(portfolio_values)
    running_max = np.maximum.accumulate(portfolio_array)
    drawdown = (portfolio_array - running_max) / running_max * 100
    
    axes[1, 1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
    axes[1, 1].plot(drawdown, color='red', linewidth=1)
    axes[1, 1].set_title('Drawdown')
    axes[1, 1].set_xlabel('Dias')
    axes[1, 1].set_ylabel('Drawdown (%)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}/evaluation_results.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Gráficos salvos em {REPORTS_DIR}/evaluation_results.png")
    plt.close()


def save_results(portfolio_values, trades_history, metrics, features_df):
    """
    Salva resultados em arquivos
    """
    # Salva evolução do portfólio
    portfolio_df = pd.DataFrame({
        'Day': range(len(portfolio_values)),
        'Portfolio_Value': portfolio_values
    })
    portfolio_df['Returns'] = portfolio_df['Portfolio_Value'].pct_change()
    portfolio_df.to_csv(f'{REPORTS_DIR}/portfolio_evolution.csv', index=False)
    
    # Salva métricas
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'{REPORTS_DIR}/metrics.csv', index=False)
    
    print(f"   ✓ Resultados salvos em {REPORTS_DIR}/")


if __name__ == "__main__":
    evaluate_agent()

