"""
Script principal - Executa treinamento e avaliação
"""

import os
from train import train_agent
from evaluate import evaluate_agent

def main():
    """
    Função principal
    """
    print("\n" + "="*70)
    print("QUANTUMFINANCE - AGENTE DE REINFORCEMENT LEARNING")
    print("="*70)
    
    # Verifica se modelo já existe
    model_path = 'models/dqn_model.pth'
    train_new = True
    
    if os.path.exists(model_path):
        resposta = input("\nModelo já existe. Deseja treinar um novo modelo? (s/n): ")
        train_new = resposta.lower() == 's'
    
    if train_new:
        print("\n>>> INICIANDO TREINAMENTO <<<")
        train_agent()
    
    print("\n>>> INICIANDO AVALIAÇÃO <<<")
    evaluate_agent()
    
    print("\n" + "="*70)
    print("PROCESSO CONCLUÍDO!")
    print("="*70)
    print("\nConsulte os relatórios em:")
    print("  - reports/evaluation_results.png")
    print("  - reports/portfolio_evolution.csv")
    print("  - reports/metrics.csv")
    print("\nConsulte também o REPORT.md para análise detalhada.\n")

if __name__ == "__main__":
    main()


