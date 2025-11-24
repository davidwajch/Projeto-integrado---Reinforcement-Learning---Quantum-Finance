# QuantumFinance - Agente de Reinforcement Learning para Trading Automatizado

## Descrição do Projeto

Este projeto implementa um agente de Reinforcement Learning (RL) usando Deep Q-Network (DQN) para operar automaticamente três ativos da bolsa brasileira:
- **Vale** (VALE3)
- **Petrobrás** (PETR4)
- **Brasil Foods** (BRFS3)

## Estrutura do Projeto

```
.
├── README.md
├── requirements.txt
├── data/                    # Dados históricos das ações
├── models/                  # Modelos treinados salvos
├── reports/                 # Relatórios e visualizações
├── src/
│   ├── __init__.py
│   ├── environment.py      # Ambiente de RL (estados, ações, recompensas)
│   ├── agent.py            # Agente DQN
│   ├── data_loader.py      # Download e processamento de dados
│   ├── metrics.py          # Métricas de avaliação financeira
│   └── utils.py            # Funções auxiliares
├── train.py                # Script de treinamento
├── evaluate.py             # Script de avaliação
└── main.py                 # Script principal

```

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

### Treinamento do Agente

```bash
python train.py
```

### Avaliação do Agente

```bash
python evaluate.py
```

### Execução Completa

```bash
python main.py
```

## Componentes Principais

### 1. Ambiente de RL (Environment)
- **Estados**: Features técnicas (preços, volumes, indicadores técnicos)
- **Ações**: Comprar, Vender ou Manter posição para cada ativo
- **Recompensas**: Baseadas no lucro/prejuízo e métricas de risco

### 2. Agente DQN
- Rede neural profunda para aproximação da função Q
- Experience Replay para estabilidade
- Target Network para convergência

### 3. Métricas de Avaliação
- Retorno total
- Sharpe Ratio
- Maximum Drawdown
- Win Rate

## Relatório

Consulte o arquivo `REPORT.md` para detalhes completos sobre o desenho do agente, resultados e insights.


