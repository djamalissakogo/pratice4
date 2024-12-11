import numpy as np
import matplotlib.pyplot as plt

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        return tuple(self.board.flatten())

    def make_move(self, row, col):
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            winner = self.check_winner()
            self.current_player = 3 - self.current_player
            return winner
        return None

    def check_winner(self):
        for i in range(3):
            if np.all(self.board[i, :] == self.current_player) or np.all(self.board[:, i] == self.current_player):
                return self.current_player
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] == self.current_player:
            return self.current_player
        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] == self.current_player:
            return self.current_player
        if np.all(self.board != 0):
            return 0
        return None

    def get_valid_actions(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def render(self):
        symbols = {0: '.', 1: 'X', 2: 'O'}
        print("\n".join(" ".join(symbols[cell] for cell in row) for row in self.board))
        print()

# Q-Learning
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, valid_actions):
        if np.random.rand() < self.epsilon:
            return valid_actions[np.random.randint(len(valid_actions))]
        q_values = [self.get_q_value(state, action) for action in valid_actions]
        max_q = max(q_values)
        return valid_actions[q_values.index(max_q)]

    def update_q_value(self, state, action, reward, next_state, next_valid_actions):
        current_q = self.get_q_value(state, action)
        max_next_q = max([self.get_q_value(next_state, a) for a in next_valid_actions], default=0.0)
        self.q_table[(state, action)] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

# Обучение
def train_agent(episodes):
    env = TicTacToe()
    agent = QLearningAgent()
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.choose_action(state, valid_actions)

            winner = env.make_move(*action)
            env.render()
            next_state = env.get_state()
            next_valid_actions = env.get_valid_actions()

            if winner is None:
                reward = 0
            elif winner == 0:
                reward = 0.5
                done = True
            elif winner == 1:
                reward = 1
                done = True
            else:
                reward = -1
                done = True

            agent.update_q_value(state, action, reward, next_state, next_valid_actions)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    return rewards

if __name__ == "__main__":
    episodes = 1000
    rewards = train_agent(episodes)

    # Changer la couleur du graphique
    plt.plot(range(episodes), rewards, color='red')
    plt.xlabel("Эпизоды")
    plt.ylabel("Награда")
    plt.title("Зависимость награды от числа эпизодов")
    plt.show()
    
    # episodes = 1000
    # rewards = train_agent(episodes)

    # plt.plot(range(episodes), rewards)
    # plt.xlabel("Эпизоды")
    # plt.ylabel("Награда")
    # plt.title("Зависимость награды от числа эпизодов")
    # plt.show()
    
    


    