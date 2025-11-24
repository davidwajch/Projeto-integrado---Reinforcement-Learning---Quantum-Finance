"""
Agente DQN (Deep Q-Network) para Reinforcement Learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Tuple, List


class DQNNetwork(nn.Module):
    """
    Rede Neural para aproximação da função Q
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int] = [128, 128, 64]):
        """
        Inicializa a rede neural
        
        Args:
            state_size: Tamanho do vetor de estado
            action_size: Número de ações possíveis
            hidden_layers: Lista com tamanhos das camadas ocultas
        """
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_size = state_size
        
        # Cria camadas ocultas
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_size = hidden_size
        
        # Camada de saída
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da rede
        
        Args:
            state: Estado atual
        
        Returns:
            Valores Q para cada ação
        """
        return self.network(state)


class DQNAgent:
    """
    Agente DQN com Experience Replay e Target Network
    """
    
    def __init__(self, state_size: int, action_size: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 100):
        """
        Inicializa o agente DQN
        
        Args:
            state_size: Tamanho do vetor de estado
            action_size: Número de ações possíveis
            learning_rate: Taxa de aprendizado
            gamma: Fator de desconto
            epsilon: Taxa de exploração inicial
            epsilon_min: Taxa de exploração mínima
            epsilon_decay: Decaimento da taxa de exploração
            memory_size: Tamanho do buffer de experiência
            batch_size: Tamanho do batch para treinamento
            target_update_freq: Frequência de atualização da target network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device (GPU se disponível)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando device: {self.device}")
        
        # Redes neurais
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience Replay Buffer
        self.memory = deque(maxlen=memory_size)
        
        # Contador de steps para atualização da target network
        self.step_count = 0
        
        # Copia pesos iniciais para target network
        self.update_target_network()
    
    def update_target_network(self):
        """
        Atualiza a target network copiando os pesos da Q-network
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """
        Armazena experiência no buffer de replay
        
        Args:
            state: Estado atual
            action: Ação executada
            reward: Recompensa recebida
            next_state: Próximo estado
            done: Se o episódio terminou
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Seleciona ação usando epsilon-greedy
        
        Args:
            state: Estado atual
            training: Se está em modo de treinamento
        
        Returns:
            Ação selecionada
        """
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.cpu().data.numpy().argmax()
    
    def replay(self) -> float:
        """
        Treina a rede usando experiência do buffer
        
        Returns:
            Loss médio do treinamento
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch aleatório
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Converte para tensores
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Q-values atuais
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Q-values do próximo estado (usando target network)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calcula loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping para estabilidade
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Atualiza epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Atualiza target network periodicamente
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()
    
    def save(self, filepath: str):
        """
        Salva o modelo
        
        Args:
            filepath: Caminho para salvar o modelo
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
        print(f"Modelo salvo em {filepath}")
    
    def load(self, filepath: str):
        """
        Carrega o modelo
        
        Args:
            filepath: Caminho do modelo
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.step_count = checkpoint.get('step_count', 0)
        print(f"Modelo carregado de {filepath}")


