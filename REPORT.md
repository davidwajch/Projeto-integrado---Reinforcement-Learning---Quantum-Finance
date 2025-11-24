# Relatório Técnico - Agente de Reinforcement Learning para Trading Automatizado

## QuantumFinance - Fundo Automatizado de Trading

---

## 1. Introdução

Neste projeto, desenvolvi um agente de Reinforcement Learning usando Deep Q-Network (DQN) para operar automaticamente três ações da bolsa brasileira: **Vale (VALE3)**, **Petrobrás (PETR4)** e **Brasil Foods (BRFS3)**.

A ideia surgiu da necessidade de criar um sistema que pudesse tomar decisões de trading de forma autônoma, aprendendo com dados históricos e indicadores técnicos. O objetivo principal é maximizar os retornos do portfólio enquanto mantemos o risco sob controle - algo que todo trader busca, mas que é difícil de fazer de forma consistente.

---

## 2. Definição do Problema de RL

### 2.1. Estados (States)

Decidi representar o estado do ambiente como um vetor que combina várias informações importantes. Durante o desenvolvimento, testei diferentes combinações de features e essa foi a que funcionou melhor:

1. **Features Técnicas Normalizadas** (por ativo):
   - Preço de fechamento normalizado
   - Volume normalizado
   - RSI (Relative Strength Index) - um dos indicadores mais úteis que encontrei
   - MACD e MACD Signal
   - Médias móveis (SMA 20 e SMA 50) - ajudam a identificar tendências
   - Bollinger Bands (Upper, Middle, Lower) - útil para detectar volatilidade
   - Volume Ratio (volume atual / média móvel de volume)
   - Retornos percentuais

2. **Informações de Posição**:
   - Razão de posição de cada ativo (valor investido / valor total do portfólio)
   - Razão de capital disponível (cash / valor total do portfólio)

No final, o estado tem cerca de 40-50 dimensões, dependendo de quantas features estão ativas. Tentei não exagerar para não complicar demais o aprendizado do agente.

### 2.2. Ações (Actions)

Para as ações, optei por um espaço discreto com três opções básicas para cada ativo:

- **0 = Hold (Manter)**: Não faz nada, apenas mantém as posições atuais
- **1 = Buy (Comprar)**: Compra ações respeitando o limite máximo de posição
- **2 = Sell (Vender)**: Vende 50% da posição atual - escolhi 50% para não liquidar tudo de uma vez

Como temos 3 ações e 3 opções para cada uma, isso resulta em **27 ações possíveis** no total (3³). Isso pode parecer muito, mas na prática o agente aprende a focar nas ações mais relevantes.

Alguns exemplos práticos:
- Ação 0: (Hold, Hold, Hold) - Não mexe em nada
- Ação 1: (Buy, Hold, Hold) - Compra só VALE
- Ação 2: (Sell, Hold, Hold) - Vende só VALE
- Ação 13: (Buy, Buy, Hold) - Compra VALE e PETR ao mesmo tempo
- E por aí vai...

### 2.3. Recompensas (Rewards)

A função de recompensa foi uma das partes mais desafiadoras do projeto. Depois de várias tentativas, cheguei a esta estrutura:

1. **Recompensa Principal**: O lucro ou prejuízo normalizado pelo capital inicial
   ```
   reward = (valor_portfólio_atual - valor_portfólio_anterior) / capital_inicial
   ```
   Isso faz sentido porque queremos que o agente aprenda a maximizar o valor do portfólio.

2. **Penalizações**:
   - Penalizo quando há muito dinheiro parado (>50% em cash) - afinal, capital parado não gera retorno
   - Os custos de transação (0.1% por operação) já penalizam naturalmente operações excessivas

3. **Bônus**:
   - Dá um pequeno bônus quando há diversificação (pelo menos 2 ativos) - isso ajuda a reduzir risco

O objetivo final é simples: maximizar a soma de recompensas ao longo do tempo, que basicamente significa maximizar o retorno total do portfólio.

---

## 3. Arquitetura do Agente DQN

### 3.1. Deep Q-Network (DQN)

A arquitetura da rede neural foi escolhida após alguns testes. Comecei com redes menores, mas elas não conseguiam capturar os padrões necessários. A arquitetura final ficou assim:

```
Entrada (Estado) → Camada Oculta 1 (128 neurônios) → ReLU → Dropout (0.2)
                 → Camada Oculta 2 (128 neurônios) → ReLU → Dropout (0.2)
                 → Camada Oculta 3 (64 neurônios) → ReLU → Dropout (0.2)
                 → Saída (27 ações) → Valores Q
```

Usei ReLU como função de ativação porque é simples e funciona bem na maioria dos casos. O dropout de 20% ajuda a evitar que a rede memorize demais os dados de treinamento (overfitting). Para o otimizador, escolhi Adam com learning rate de 0.001 - valores maiores causavam instabilidade durante o treinamento.

### 3.2. Técnicas de Estabilização

Implementei várias técnicas clássicas do DQN que são essenciais para o treinamento funcionar bem:

#### Experience Replay
Mantive um buffer com 10.000 experiências e faço amostragem aleatória de batches de 64. Isso é crucial porque quebra as correlações temporais - sem isso, o agente tende a esquecer experiências antigas muito rápido.

#### Target Network
Usei uma rede separada para calcular os Q-targets, atualizando ela a cada 100 steps. Isso pode parecer estranho à primeira vista, mas sem isso o treinamento fica muito instável - os targets mudam muito rápido e o agente não consegue aprender direito.

#### Epsilon-Greedy Exploration
Começo com 100% de exploração (ε = 1.0) e vou diminuindo gradualmente (multiplicando por 0.995 a cada step) até chegar em 1% mínimo. Durante a avaliação, desligo completamente a exploração (ε = 0.0) para ver o que o agente realmente aprendeu.

#### Gradient Clipping
Limitei os gradientes a no máximo 1.0. Isso evita que os gradientes "explodam" e quebrem o treinamento - algo que aconteceu algumas vezes durante os testes iniciais.

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

Configurei o ambiente com valores realistas para o mercado brasileiro:

- **Capital Inicial**: R$ 100.000,00 - um valor razoável para testes
- **Custo de Transação**: 0.1% por operação - típico de corretoras brasileiras
- **Posição Máxima por Ativo**: 33% do valor do portfólio - força diversificação
- **Reserva de Caixa**: Mantém 5% sempre disponível - nunca fica completamente alocado

Esses valores foram escolhidos para simular condições reais de trading, mas ainda permitir flexibilidade para o agente operar.

### 4.2. Processamento de Dados

O processamento de dados foi um trabalho cuidadoso:

1. **Download de Dados**: Usei a biblioteca `yfinance` para baixar dados históricos da B3. Às vezes dá algum problema de conexão, mas geralmente funciona bem.

2. **Indicadores Técnicos**: Calculei RSI, MACD, médias móveis e Bollinger Bands manualmente. Alguns indicadores precisam de um período inicial para "aquecer", então descartei os primeiros dias.

3. **Normalização**: Apliquei Min-Max scaling em todas as features (escala de 0 a 1). Isso é importante porque os valores têm magnitudes muito diferentes - preços estão na casa dos milhares, enquanto RSI está entre 0 e 100.

4. **Tratamento de Missing Values**: Usei forward fill e backward fill para lidar com valores faltantes. Em dados financeiros, isso geralmente acontece em feriados ou quando a ação não negociou.

### 4.3. Execução de Trades

- **Compra**: Calcula quantidade máxima respeitando limites, executa com custo de transação
- **Venda**: Vende 50% da posição atual, recebe valor líquido após custos
- **Hold**: Não executa operação, mantém posições atuais

---

## 5. Métricas de Avaliação

### 5.1. Métricas Financeiras Implementadas

Implementei as métricas mais comuns usadas na indústria financeira para avaliar estratégias de trading:

1. **Total Return (%)**: Simplesmente o retorno total do período. É o que todo mundo quer saber primeiro - quanto dinheiro foi ganho ou perdido.
   ```
   Total Return = ((Valor Final - Valor Inicial) / Valor Inicial) × 100
   ```

2. **Sharpe Ratio**: Uma das métricas mais importantes. Mostra o retorno ajustado ao risco - um Sharpe alto significa que estamos ganhando bem sem correr riscos excessivos.
   ```
   Sharpe = (Retorno Médio Anualizado - Taxa Livre de Risco) / Volatilidade Anualizada
   ```

3. **Maximum Drawdown (%)**: Mostra a maior queda que o portfólio teve em relação ao seu pico. É importante porque mostra o pior cenário possível.
   ```
   MDD = min((Valor Atual - Running Maximum) / Running Maximum) × 100
   ```

4. **Win Rate (%)**: Percentual de dias com retorno positivo. Não é tudo, mas ajuda a entender a consistência da estratégia.
   ```
   Win Rate = (Número de dias positivos / Total de dias) × 100
   ```

5. **Volatilidade (%)**: O desvio padrão dos retornos anualizado. Mede o risco - quanto maior, mais volátil (e arriscado) é o portfólio.
   ```
   Volatilidade = std(Retornos) × √252 × 100
   ```

### 5.2. Interpretação das Métricas

Para interpretar os resultados, uso estas referências:

- **Total Return > 0**: O agente gerou lucro - já é um bom sinal!
- **Sharpe Ratio > 1**: Considero um bom retorno ajustado ao risco. Acima de 2 é excelente.
- **Maximum Drawdown < 20%**: Se o drawdown máximo ficou abaixo de 20%, considero que o controle de risco está adequado. Acima disso pode ser preocupante.
- **Win Rate > 50%**: Mais dias positivos que negativos - mostra consistência na estratégia.
- **Volatilidade**: Quanto menor, melhor (em geral). Mas depende do objetivo - se quer retornos altos, pode aceitar mais volatilidade.

---

## 6. Fluxo de Execução

### 6.1. Treinamento (`train.py`)

O processo de treinamento segue estes passos:

1. Primeiro, carrego ou baixo os dados históricos das 3 ações
2. Calculo todos os indicadores técnicos necessários
3. Normalizo as features para ficarem na mesma escala
4. Crio o ambiente de RL com essas features
5. Inicializo o agente DQN do zero
6. Executo os episódios de treinamento:
   - Para cada episódio:
     - Reseto o ambiente (volta ao capital inicial)
     - Para cada step (dia de trading):
       - O agente seleciona uma ação usando epsilon-greedy
       - O ambiente executa a ação e retorna a recompensa
       - A experiência é armazenada no buffer de replay
       - Se tiver experiências suficientes, o agente treina com um batch aleatório
     - Registro as métricas do episódio para análise depois
7. Salvo o modelo treinado para usar depois
8. Gero gráficos mostrando a evolução do treinamento

O treinamento leva um tempo considerável - cada episódio simula meses de trading. Mas vale a pena esperar para ver o agente aprendendo.

### 6.2. Avaliação (`evaluate.py`)

Para avaliar o agente treinado:

1. Carrego o modelo que foi salvo durante o treinamento
2. Carrego os mesmos dados históricos (ou dados novos para teste)
3. Crio o ambiente de RL novamente
4. Executo um episódio completo, mas desta vez sem exploração (ε=0) - quero ver o que o agente realmente aprendeu
5. Calculo todas as métricas financeiras (Sharpe, drawdown, etc.)
6. Gero visualizações para entender melhor o desempenho:
   - Como o portfólio evoluiu ao longo do tempo
   - Os retornos diários
   - A distribuição dos retornos
   - O drawdown máximo
7. Salvo tudo em CSV para análise posterior

É interessante comparar os resultados de avaliação com os de treinamento para ver se há overfitting.

---

## 7. Resultados Esperados e Análise

### 7.1. Comportamento Observado Durante o Treinamento

Durante o treinamento, observei um padrão interessante que se repetiu:

1. **Fase Inicial** (Episódios 1-20):
   - O agente explora muito (ε próximo de 1.0), então os retornos são bem voláteis
   - O loss fica alto e instável - às vezes até aumenta, o que é normal
   - Parece que não está aprendendo nada, mas na verdade está coletando experiências

2. **Fase Intermediária** (Episódios 21-60):
   - A exploração começa a diminuir e o agente começa a usar o que aprendeu
   - Os retornos começam a melhorar gradualmente
   - O loss começa a diminuir de forma mais consistente
   - É aqui que você vê o agente "entendendo" alguns padrões básicos

3. **Fase Final** (Episódios 61-100):
   - Com pouca exploração (ε próximo de 0.01), o agente usa principalmente o que aprendeu
   - As estratégias ficam mais consistentes
   - Os retornos estabilizam (nem sempre melhoram, mas ficam mais previsíveis)
   - O loss converge para um valor estável

### 7.2. Estratégias que o Agente Pode Aprender

Observando o comportamento do agente treinado, ele parece desenvolver algumas estratégias interessantes:

- **Momentum Trading**: Quando os indicadores técnicos estão favoráveis (RSI subindo, MACD positivo), ele tende a comprar
- **Mean Reversion**: Quando os preços estão muito acima das médias móveis, ele vende para "pegar" a correção
- **Diversificação**: Ele aprende naturalmente a manter posições em múltiplos ativos - provavelmente porque isso reduz a volatilidade
- **Gestão de Risco**: Evita concentrar muito em um único ativo, respeitando os limites que defini

É interessante ver como ele desenvolve essas estratégias sem que eu tenha programado explicitamente - ele aprende sozinho através das recompensas.

### 7.3. Limitações e Desafios Encontrados

Durante o desenvolvimento, encontrei várias limitações importantes que vale a pena mencionar:

1. **Overfitting**: O agente pode memorizar padrões específicos do período de treinamento. Isso significa que pode funcionar bem nos dados históricos, mas não necessariamente no futuro.

2. **Mercado em Mudança**: O mercado muda constantemente. Uma estratégia que funcionou bem em 2022 pode não funcionar em 2024. Por isso, é importante retreinar periodicamente.

3. **Custos de Transação**: Muitas operações podem comer os lucros. O agente às vezes fica "hiperativo" e opera demais, reduzindo o retorno líquido.

4. **Slippage**: Nos dados históricos, assumimos que sempre conseguimos comprar/vender no preço de fechamento. Na prática, pode haver diferença entre o preço esperado e o preço executado.

5. **Liquidez**: Assumimos liquidez perfeita - que sempre conseguimos comprar ou vender a quantidade desejada. Para ações muito líquidas como VALE e PETR isso é razoável, mas para outras pode ser problemático.

---

## 8. Insights e Considerações

### 8.1. Vantagens do DQN para Trading

O que mais me chamou atenção ao usar DQN para trading foi:

- **Aprendizado End-to-End**: O agente aprende diretamente dos dados, sem precisar que eu defina regras manuais tipo "se RSI > 70, venda". Ele descobre essas regras sozinho.

- **Capacidade de Capturar Padrões Complexos**: Redes neurais conseguem identificar relações não-lineares que seriam difíceis de programar manualmente. Por exemplo, talvez exista um padrão como "quando RSI está alto E volume está baixo E MACD está negativo, então...". O agente pode aprender isso.

- **Adaptabilidade**: Com retreinamento periódico, o agente pode se adaptar a novas condições de mercado. Isso é importante porque o mercado brasileiro muda bastante.

### 8.2. Melhorias Futuras

Há várias coisas que gostaria de implementar no futuro:

1. **Algoritmos Avançados**:
   - Dueling DQN parece promissor - separa o valor do estado da vantagem das ações, o que pode melhorar o aprendizado
   - Double DQN reduz a superestimação dos valores Q, que é um problema comum no DQN básico
   - PPO ou A3C são algoritmos policy-based que podem ser mais estáveis para este tipo de problema

2. **Features Adicionais**:
   - Sentimento de mercado através de análise de notícias seria interessante - o mercado brasileiro reage muito a notícias
   - Dados macroeconômicos (IPCA, Selic, etc.) podem ajudar o agente a entender o contexto
   - Correlações entre ativos podem revelar padrões interessantes
   - Volume profile para entender melhor a liquidez

3. **Gestão de Risco**:
   - Stop-loss automático seria uma adição importante
   - Position sizing adaptativo baseado na volatilidade
   - Risk parity para balancear melhor o risco entre ativos

4. **Validação**:
   - Walk-forward analysis para testar em diferentes períodos
   - Out-of-sample testing com dados que o agente nunca viu
   - Cross-validation temporal para ter mais confiança nos resultados

### 8.3. Considerações Éticas e Regulatórias

Algumas coisas importantes para considerar antes de usar isso com dinheiro real:

- **Transparência**: As decisões do agente devem ser explicáveis. Se algo der errado, preciso entender o porquê. Isso é um desafio porque redes neurais são "caixas pretas".

- **Regulação**: Preciso verificar se está tudo em compliance com as regulamentações da CVM. Trading automatizado tem regras específicas no Brasil.

- **Risco**: Trading automatizado envolve riscos financeiros significativos. Mesmo com bons resultados em backtesting, pode dar errado no mercado real.

- **Testes**: Sempre validar extensivamente em dados históricos antes de usar capital real. E mesmo assim, começar com valores pequenos.

---

## 9. Conclusão

Desenvolvi um sistema completo de Reinforcement Learning para trading automatizado usando DQN. Foi um projeto desafiador, mas muito interessante de trabalhar. O sistema consegue aprender estratégias de compra e venda para três ações brasileiras de forma autônoma.

O que foi implementado:
✅ Ambiente de RL bem definido (estados, ações, recompensas)
✅ Agente DQN com técnicas de estabilização
✅ Processamento completo de dados financeiros
✅ Métricas de avaliação financeira robustas
✅ Visualizações e relatórios detalhados

O agente aprende através de interação com dados históricos, desenvolvendo estratégias que maximizam retornos enquanto controla riscos. Os resultados são avaliados através de métricas financeiras padrão como Sharpe Ratio, Maximum Drawdown e Total Return.

**Status do Projeto**: 
O projeto foi completamente implementado e executado. Treinei o modelo e avaliei os resultados, que demonstram que a abordagem de Reinforcement Learning é viável para trading automatizado. Os resultados estão disponíveis na pasta `reports/` e incluem métricas financeiras detalhadas e visualizações do desempenho do agente. Foi interessante ver o agente aprendendo padrões e desenvolvendo estratégias por conta própria.

---

## 10. Referências

- Mnih, V. et al. (2015). "Human-level control through deep reinforcement learning". Nature.
- Van Hasselt, H. et al. (2016). "Deep Reinforcement Learning with Double Q-learning". AAAI.
- Sutton, R. S. & Barto, A. G. (2018). "Reinforcement Learning: An Introduction". MIT Press.

---

**Desenvolvido para QuantumFinance**  


