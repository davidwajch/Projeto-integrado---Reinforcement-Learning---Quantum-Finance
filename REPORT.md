# Relatório Técnico - Agente de Reinforcement Learning para Trading Automatizado

## QuantumFinance - Fundo Automatizado de Trading

---

## 1. Introdução

Este projeto implementa um agente de Reinforcement Learning (RL) utilizando Deep Q-Network (DQN) para operar automaticamente três ativos da bolsa brasileira: **Vale (VALE3)**, **Petrobrás (PETR4)** e **Brasil Foods (BRFS3)**.

O objetivo é desenvolver um sistema automatizado capaz de tomar decisões financeiras inteligentes (comprar, vender ou manter posição) baseado em dados históricos e indicadores técnicos, maximizando o retorno do portfólio enquanto controla o risco.

---

## 2. Definição do Problema de RL

### 2.1. Estados (States)

O estado do ambiente é representado por um vetor contendo:

1. **Features Técnicas Normalizadas** (por ativo):
   - Preço de fechamento normalizado
   - Volume normalizado
   - RSI (Relative Strength Index)
   - MACD e MACD Signal
   - Médias móveis (SMA 20 e SMA 50)
   - Bollinger Bands (Upper, Middle, Lower)
   - Volume Ratio (volume atual / média móvel de volume)
   - Retornos percentuais

2. **Informações de Posição**:
   - Razão de posição de cada ativo (valor investido / valor total do portfólio)
   - Razão de capital disponível (cash / valor total do portfólio)

**Tamanho do Estado**: Variável dependendo do número de features, aproximadamente 40-50 dimensões.

### 2.2. Ações (Actions)

O espaço de ações é definido como combinações de ações para os três ativos:

- **0 = Hold (Manter)**: Não realiza operação
- **1 = Buy (Comprar)**: Compra ações respeitando limite de posição máxima
- **2 = Sell (Vender)**: Vende 50% da posição atual

**Total de Ações**: 3³ = **27 ações possíveis** (combinações de ações para VALE, PETR e BRFS)

**Exemplo de ações**:
- Ação 0: (Hold, Hold, Hold) - Não opera nenhum ativo
- Ação 1: (Buy, Hold, Hold) - Compra VALE apenas
- Ação 2: (Sell, Hold, Hold) - Vende VALE apenas
- Ação 13: (Buy, Buy, Hold) - Compra VALE e PETR
- E assim por diante...

### 2.3. Recompensas (Rewards)

A função de recompensa é projetada para:

1. **Recompensa Principal**: Lucro/prejuízo normalizado pela capital inicial
   ```
   reward = (valor_portfólio_atual - valor_portfólio_anterior) / capital_inicial
   ```

2. **Penalizações**:
   - Penalização por manter muito capital parado (>50% em cash)
   - Penalização implícita por custos de transação (0.1% por operação)

3. **Bônus**:
   - Bônus por diversificação (ter posições em pelo menos 2 ativos)

**Objetivo**: Maximizar a soma de recompensas ao longo do tempo, equivalente a maximizar o retorno total do portfólio.

---

## 3. Arquitetura do Agente DQN

### 3.1. Deep Q-Network (DQN)

O agente utiliza uma arquitetura de rede neural profunda para aproximar a função Q (Quality):

```
Entrada (Estado) → Camada Oculta 1 (128 neurônios) → ReLU → Dropout (0.2)
                 → Camada Oculta 2 (128 neurônios) → ReLU → Dropout (0.2)
                 → Camada Oculta 3 (64 neurônios) → ReLU → Dropout (0.2)
                 → Saída (27 ações) → Valores Q
```

**Função de Ativação**: ReLU (Rectified Linear Unit)
**Regularização**: Dropout de 20% para evitar overfitting
**Otimizador**: Adam com learning rate de 0.001

### 3.2. Técnicas de Estabilização

#### Experience Replay
- Buffer de memória com 10.000 experiências
- Amostragem aleatória de batches de 64 experiências
- Quebra correlações temporais e melhora estabilidade

#### Target Network
- Rede separada para cálculo de Q-targets
- Atualizada a cada 100 steps da Q-network principal
- Reduz instabilidade durante treinamento

#### Epsilon-Greedy Exploration
- Exploração inicial: ε = 1.0 (100% aleatório)
- Decaimento: ε *= 0.995 a cada step
- Mínimo: ε_min = 0.01 (1% exploração)
- Durante avaliação: ε = 0.0 (apenas exploração)

#### Gradient Clipping
- Limita gradientes a máximo de 1.0
- Previne explosão de gradientes

### 3.3. Hiperparâmetros

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| Learning Rate | 0.001 | Taxa de aprendizado |
| Gamma (γ) | 0.95 | Fator de desconto futuro |
| Epsilon inicial | 1.0 | Taxa de exploração inicial |
| Epsilon mínimo | 0.01 | Taxa de exploração mínima |
| Epsilon decay | 0.995 | Decaimento da exploração |
| Memory size | 10.000 | Tamanho do buffer de replay |
| Batch size | 64 | Tamanho do batch de treinamento |
| Target update freq | 100 | Frequência de atualização da target network |

---

## 4. Ambiente de Trading

### 4.1. Configurações do Ambiente

- **Capital Inicial**: R$ 100.000,00
- **Custo de Transação**: 0.1% por operação (compra ou venda)
- **Posição Máxima por Ativo**: 33% do valor do portfólio
- **Reserva de Caixa**: Mantém 5% do capital disponível como reserva

### 4.2. Processamento de Dados

1. **Download de Dados**: Utiliza biblioteca `yfinance` para baixar dados históricos da B3
2. **Indicadores Técnicos**: Calcula RSI, MACD, médias móveis, Bollinger Bands
3. **Normalização**: Min-Max scaling para todas as features (0 a 1)
4. **Tratamento de Missing Values**: Forward fill e backward fill

### 4.3. Execução de Trades

- **Compra**: Calcula quantidade máxima respeitando limites, executa com custo de transação
- **Venda**: Vende 50% da posição atual, recebe valor líquido após custos
- **Hold**: Não executa operação, mantém posições atuais

---

## 5. Métricas de Avaliação

### 5.1. Métricas Financeiras Implementadas

1. **Total Return (%)**: Retorno total do período
   ```
   Total Return = ((Valor Final - Valor Inicial) / Valor Inicial) × 100
   ```

2. **Sharpe Ratio**: Retorno ajustado ao risco
   ```
   Sharpe = (Retorno Médio Anualizado - Taxa Livre de Risco) / Volatilidade Anualizada
   ```

3. **Maximum Drawdown (%)**: Maior queda do portfólio em relação ao pico
   ```
   MDD = min((Valor Atual - Running Maximum) / Running Maximum) × 100
   ```

4. **Win Rate (%)**: Percentual de dias com retorno positivo
   ```
   Win Rate = (Número de dias positivos / Total de dias) × 100
   ```

5. **Volatilidade (%)**: Desvio padrão dos retornos anualizado
   ```
   Volatilidade = std(Retornos) × √252 × 100
   ```

### 5.2. Interpretação das Métricas

- **Total Return > 0**: Agente gerou lucro
- **Sharpe Ratio > 1**: Bom retorno ajustado ao risco
- **Maximum Drawdown < 20%**: Controle de risco adequado
- **Win Rate > 50%**: Mais dias positivos que negativos
- **Volatilidade**: Medida de risco do portfólio

---

## 6. Fluxo de Execução

### 6.1. Treinamento (`train.py`)

1. Carrega/baixa dados históricos das 3 ações
2. Calcula indicadores técnicos
3. Normaliza features
4. Cria ambiente de RL
5. Inicializa agente DQN
6. Executa N episódios de treinamento:
   - Para cada episódio:
     - Reseta ambiente
     - Para cada step:
       - Agente seleciona ação (epsilon-greedy)
       - Ambiente executa ação e retorna recompensa
       - Experiência armazenada no buffer
       - Agente treina com batch aleatório
     - Registra métricas do episódio
7. Salva modelo treinado
8. Gera gráficos de treinamento

### 6.2. Avaliação (`evaluate.py`)

1. Carrega modelo treinado
2. Carrega dados históricos
3. Cria ambiente de RL
4. Executa um episódio completo sem exploração (ε=0)
5. Calcula métricas financeiras
6. Gera visualizações:
   - Evolução do portfólio
   - Retornos diários
   - Distribuição de retornos
   - Drawdown
7. Salva resultados em CSV

---

## 7. Resultados Esperados e Análise

### 7.1. Comportamento Esperado do Agente

Durante o treinamento, espera-se observar:

1. **Fase Inicial** (Episódios 1-20):
   - Alta exploração (ε próximo de 1.0)
   - Retornos voláteis e imprevisíveis
   - Loss alto e instável

2. **Fase Intermediária** (Episódios 21-60):
   - Redução gradual da exploração
   - Aprendizado de padrões básicos
   - Melhoria gradual nos retornos
   - Loss diminuindo

3. **Fase Final** (Episódios 61-100):
   - Baixa exploração (ε próximo de 0.01)
   - Estratégias mais consistentes
   - Retornos mais estáveis
   - Loss convergindo

### 7.2. Estratégias que o Agente Pode Aprender

- **Momentum Trading**: Comprar quando indicadores técnicos são favoráveis
- **Mean Reversion**: Vender quando preços estão muito acima da média
- **Diversificação**: Manter posições em múltiplos ativos
- **Gestão de Risco**: Evitar posições muito concentradas

### 7.3. Limitações e Desafios

1. **Overfitting**: Agente pode memorizar padrões específicos do período de treinamento
2. **Mercado em Mudança**: Estratégias aprendidas podem não funcionar em novos regimes de mercado
3. **Custos de Transação**: Muitas operações podem reduzir lucros líquidos
4. **Slippage**: Preços reais podem diferir dos preços históricos
5. **Liquidez**: Assumimos liquidez perfeita (sempre possível comprar/vender)

---

## 8. Insights e Considerações

### 8.1. Vantagens do DQN para Trading

- **Aprendizado End-to-End**: Aprende diretamente dos dados sem regras manuais
- **Capacidade de Capturar Padrões Complexos**: Redes neurais podem identificar relações não-lineares
- **Adaptabilidade**: Pode se adaptar a diferentes condições de mercado (com retreinamento)

### 8.2. Melhorias Futuras

1. **Algoritmos Avançados**:
   - Dueling DQN: Separa valor do estado e vantagem das ações
   - Double DQN: Reduz overestimation de valores Q
   - PPO ou A3C: Algoritmos policy-based mais estáveis

2. **Features Adicionais**:
   - Sentimento de mercado (análise de notícias)
   - Dados macroeconômicos
   - Correlações entre ativos
   - Volume profile

3. **Gestão de Risco**:
   - Stop-loss automático
   - Position sizing adaptativo
   - Risk parity

4. **Validação**:
   - Walk-forward analysis
   - Out-of-sample testing
   - Cross-validation temporal

### 8.3. Considerações Éticas e Regulatórias

- **Transparência**: Decisões do agente devem ser explicáveis
- **Regulação**: Verificar compliance com regulamentações da CVM
- **Risco**: Trading automatizado envolve riscos financeiros significativos
- **Testes**: Sempre validar em dados históricos antes de usar capital real

---

## 9. Conclusão

Este projeto implementa um sistema completo de Reinforcement Learning para trading automatizado, utilizando DQN para aprender estratégias de compra e venda de três ativos brasileiros. O sistema inclui:

✅ Ambiente de RL bem definido (estados, ações, recompensas)
✅ Agente DQN com técnicas de estabilização
✅ Processamento completo de dados financeiros
✅ Métricas de avaliação financeira robustas
✅ Visualizações e relatórios detalhados

O agente aprende através de interação com dados históricos, desenvolvendo estratégias que maximizam retornos enquanto controla riscos. Os resultados podem ser avaliados através de métricas financeiras padrão como Sharpe Ratio, Maximum Drawdown e Total Return.

**Próximos Passos**:
1. Executar treinamento com `python train.py`
2. Avaliar desempenho com `python evaluate.py`
3. Analisar resultados e ajustar hiperparâmetros se necessário
4. Considerar melhorias futuras mencionadas acima

---

## 10. Referências

- Mnih, V. et al. (2015). "Human-level control through deep reinforcement learning". Nature.
- Van Hasselt, H. et al. (2016). "Deep Reinforcement Learning with Double Q-learning". AAAI.
- Sutton, R. S. & Barto, A. G. (2018). "Reinforcement Learning: An Introduction". MIT Press.

---

**Desenvolvido para QuantumFinance**  
**Data de Entrega: 24/11/2025**


