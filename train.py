"""
Script de treinamento do agente DQN
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import DataLoader
from src.environment import TradingEnvironment
from src.agent import DQNAgent
from src.metrics import FinancialMetrics

# Configurações
EPISODES = 100
INITIAL_BALANCE = 100000.0
MODEL_DIR = 'models'
DATA_DIR = 'data'

def train_agent():
    """
    Função principal de treinamento
    """
    print("="*60)
    print("TREINAMENTO DO AGENTE DQN")
    print("="*60)
    
    # Carrega dados
    print("\n1. Carregando dados históricos...")
    data_loader = DataLoader(data_dir=DATA_DIR)
    
    try:
        # Tenta carregar dados salvos primeiro
        data_dict = {}
        for ticker in ['VALE', 'PETR', 'BRFS']:
            try:
                df = data_loader.load_from_file(ticker)
                from src.utils import calculate_technical_indicators
                df = calculate_technical_indicators(df)
                data_dict[ticker] = df
                print(f"   ✓ Dados de {ticker} carregados do arquivo")
            except FileNotFoundError:
                print(f"   ⚠ Arquivo não encontrado para {ticker}, baixando dados...")
                df = data_loader.download_data(ticker, period='2y')
                from src.utils import calculate_technical_indicators
                df = calculate_technical_indicators(df)
                data_dict[ticker] = df
    except Exception as e:
        print(f"   ⚠ Erro ao carregar dados salvos: {e}")
        print("   Baixando dados novos...")
        data_dict = data_loader.load_all_data(period='2y')
    
    # Prepara features
    print("\n2. Preparando features...")
    features_df, prices_df = data_loader.prepare_features(data_dict)
    print(f"   ✓ Features preparadas: {len(features_df)} registros")
    print(f"   ✓ Número de features: {len(features_df.columns) - 1}")
    
    # Cria ambiente
    print("\n3. Criando ambiente de RL...")
    env = TradingEnvironment(features_df, initial_balance=INITIAL_BALANCE, original_prices=prices_df)
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    print(f"   ✓ Estado: {state_size} dimensões")
    print(f"   ✓ Ações: {action_size} possíveis")
    
    # Cria agente
    print("\n4. Inicializando agente DQN...")
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    print(f"   ✓ Agente inicializado")
    
    # Treinamento
    print(f"\n5. Iniciando treinamento ({EPISODES} episódios)...")
    print("-"*60)
    
    scores = []
    portfolio_values_history = []
    losses = []
    
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        episode_portfolio_values = [env.portfolio_value]
        episode_losses = []
        
        done = False
        step = 0
        
        while not done:
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            episode_portfolio_values.append(info['portfolio_value'])
            step += 1
            
            # Treina o agente
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                if loss > 0:
                    episode_losses.append(loss)
        
        scores.append(total_reward)
        portfolio_values_history.append(episode_portfolio_values)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses.append(avg_loss)
        
        # Métricas do episódio
        final_value = info['portfolio_value']
        total_return = FinancialMetrics.total_return(INITIAL_BALANCE, final_value)
        
        # Print progresso
        if (episode + 1) % 10 == 0 or episode == 0:
            print(f"Episódio {episode + 1}/{EPISODES} | "
                  f"Retorno: {total_return:.2f}% | "
                  f"Reward: {total_reward:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.6f}")
    
    print("-"*60)
    print("\n6. Treinamento concluído!")
    
    # Salva modelo
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'dqn_model.pth')
    agent.save(model_path)
    
    # Plota resultados
    print("\n7. Gerando gráficos...")
    plot_training_results(scores, portfolio_values_history, losses)
    
    # Métricas finais
    print("\n8. Métricas de treinamento:")
    final_portfolio_values = [ep[-1] for ep in portfolio_values_history]
    metrics = FinancialMetrics.calculate_all_metrics(
        final_portfolio_values,
        initial_value=INITIAL_BALANCE
    )
    FinancialMetrics.print_metrics(metrics)
    
    print("\n✓ Treinamento finalizado com sucesso!")
    print(f"✓ Modelo salvo em: {model_path}")
    
    return agent, env


def plot_training_results(scores, portfolio_values_history, losses):
    """
    Plota resultados do treinamento
    """
    os.makedirs('reports', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Scores por episódio
    axes[0, 0].plot(scores)
    axes[0, 0].set_title('Reward Total por Episódio')
    axes[0, 0].set_xlabel('Episódio')
    axes[0, 0].set_ylabel('Reward Total')
    axes[0, 0].grid(True)
    
    # Retorno final por episódio
    final_returns = []
    for ep_values in portfolio_values_history:
        if len(ep_values) > 1:
            ret = FinancialMetrics.total_return(ep_values[0], ep_values[-1])
            final_returns.append(ret)
    
    axes[0, 1].plot(final_returns)
    axes[0, 1].set_title('Retorno Total por Episódio')
    axes[0, 1].set_xlabel('Episódio')
    axes[0, 1].set_ylabel('Retorno (%)')
    axes[0, 1].grid(True)
    
    # Loss por episódio
    axes[1, 0].plot(losses)
    axes[1, 0].set_title('Loss Médio por Episódio')
    axes[1, 0].set_xlabel('Episódio')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)
    
    # Evolução do portfólio no último episódio
    if portfolio_values_history:
        axes[1, 1].plot(portfolio_values_history[-1])
        axes[1, 1].set_title('Evolução do Portfólio (Último Episódio)')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Valor do Portfólio (R$)')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('reports/training_results.png', dpi=300, bbox_inches='tight')
    print("   ✓ Gráficos salvos em reports/training_results.png")
    plt.close()


if __name__ == "__main__":
    train_agent()

