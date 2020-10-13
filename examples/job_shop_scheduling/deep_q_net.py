"""
This file implements the DQN agent (built by keras and torch, respectively) for the job shop scheduling problem.
    Author: hliangzhao@zju.edu.cn (http://hliangzhao.me)
"""
import random
import numpy as np
from collections import deque
from keras import layers, models, optimizers
import torch
from torch import nn, optim

from examples.job_shop_scheduling import env
from nn import mlp

EPISODES_NUM = 10000


class DQNAgent:
    """
    A DQN agent built by keras.
    """
    def __init__(self, state_size, action_size, job_num, feature_num, replay_size):
        self.state_size = state_size
        self.action_size = action_size
        self.job_num = job_num
        self.feature_num = feature_num

        self.mem = deque(maxlen=2000)           # experience pool size
        self.replay_size = replay_size          # experience replay size

        self.gamma = 0.95                       # discount
        self.eps = 0.9                          # exploration
        self.eps_min = 0.01
        self.eps_decay = 0.995
        self.lr = 0.0005
        self.net = self.get_net()

    def get_net(self):
        """
        Implement the net illustrated by ../figs/JSSP_model.png.
        Input of the net has the shape [array([[x, x]]), array([[x, x]]), ..., array([[x, x]])].
        """
        # get basic model
        basic_model = models.Sequential(name='basic_model')
        basic_model.add(layers.Dense(units=24, input_dim=self.feature_num, activation='relu'))
        basic_model.add(layers.Dense(units=24, input_dim=self.feature_num, activation='relu'))
        basic_model.add(layers.Dense(units=24, input_dim=self.feature_num, activation='relu'))
        basic_model.add(layers.Dense(units=24, input_dim=self.feature_num, activation='relu'))
        basic_model.add(layers.Dense(units=1, activation='linear'))
        basic_model.compile(loss='mse', optimizer=optimizers.Adam(lr=self.lr))

        input_list, output_list = [], []
        for i in range(self.job_num):
            input_list.append(models.Input(shape=(self.feature_num,)))
            output_list.append(basic_model(input_list[i]))
        cat = layers.concatenate(output_list)
        out = layers.Dense(self.action_size, activation='linear')(cat)
        net = models.Model(input_list, out)
        net.compile(loss='mse', optimizer=optimizers.Adam(lr=self.lr))
        return net

    def remember(self, state, action, reward, next_state, done):
        """
        Add experience into the memory pool.
        """
        self.mem.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        The eps-greedy policy.
        """
        if np.random.rand() <= self.eps:
            return random.randrange(self.action_size)
        return np.argmax(self.net.predict(state)[0])

    def experience_replay(self):
        """
        Replay the past ten experiences for model training.
        """
        minibatch = random.sample(self.mem, self.replay_size)
        for s, a, r, n_s, done in minibatch:
            if done:
                y = r
            else:
                # y is obtained by the target net
                y = r + self.gamma * np.amax(self.net.predict(n_s)[0])
            target_Q_values = self.net.predict(s)
            target_Q_values[0][a] = y
            self.net.fit(x=s, y=target_Q_values, epochs=1, verbose=0)
        # decrease eps
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay


class DQNAgent1:
    """
    A DQN agent built by torch.
    """
    def __init__(self, state_size, action_size, job_num, feature_num, replay_size):
        self.state_size = state_size
        self.action_size = action_size
        self.job_num = job_num
        self.feature_num = feature_num

        self.mem = deque(maxlen=2000)      # experience pool size
        self.replay_size = replay_size     # experience replay size

        self.gamma = 0.95                  # discount
        self.eps = 0.9                     # exploration
        self.eps_min = 0.01
        self.eps_decay = 0.995
        self.net = self.get_net()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.0005)
        self.loss = nn.MSELoss()

    def get_net(self):
        """
        Actually this net performs poorly.
        :return:
        """
        return mlp.DropoutMLP(self.job_num * self.feature_num, 256, 64, self.action_size,
                              dropout_prob1=0.5, dropout_prob2=0.5)

    def remember(self, state, action, reward, next_state, done):
        """
        Add experience into the memory pool.
        """
        self.mem.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        The eps-greedy policy.
        """
        # transfer state as the net input
        state = torch.tensor(state, dtype=torch.float).view(1, -1)
        if np.random.rand() <= self.eps:
            return random.randrange(self.action_size)
        preds = self.net(state).detach().numpy()
        return np.argmax(preds.reshape(1, -1)[0])

    def experience_replay(self):
        """
        Replay the past ten experiences for model training.
        """
        minibatch = random.sample(self.mem, self.replay_size)
        for s, a, r, n_s, done in minibatch:
            s_in = torch.tensor(s, dtype=torch.float).view(1, -1)
            n_s_in = torch.tensor(n_s, dtype=torch.float).view(1, -1)
            self.net.eval()
            if done:
                y = r
            else:
                # y is obtained by the target net
                preds = self.net(n_s_in).detach().numpy()
                y = r + self.gamma * np.amax(preds.reshape(1, -1)[0])
            preds = self.net(s_in).detach().numpy()
            target_Q_values = preds.reshape(1, -1)[0]
            target_Q_values[a] = np.float32(y)

            # train
            self.net.train()
            y_hat = self.net(s_in).float()
            ls = self.loss(y_hat, torch.tensor(target_Q_values, dtype=torch.float).view(1, -1)).sum()
            self.optimizer.zero_grad()
            ls.backward()
            self.optimizer.step()
        # decrease eps
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay


def train(agent, job_num, machine_num, episode_num=EPISODES_NUM):
    succ_count = 0
    for e in range(episode_num):
        print('current episode: ', e)
        scenario = env.Env(machine_num, job_num)
        state, score, done = scenario.step()
        action_list, old_score, score = [], 0, 0

        for m in range(job_num * machine_num):
            action = agent.act(state)
            next_state, score, done = scenario.step(action)
            reward = old_score - score + 15 if not done else -10000
            old_score = score
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                if m >= job_num * machine_num - 1:
                    succ_count += 1
                # if not all jobs finished but we get an wrong action, then this game is done
                break

            action_list.append(action)

        if len(agent.mem) > agent.replay_size:
            agent.experience_replay()

        if e % 100 == 0:
            print('Episode: {}/{}, makespan: {}, success: {}/100, eps: {:.2}'.format(e, episode_num, score, succ_count, agent.eps))
            print(action_list, len(action_list))
            with open('log', 'a') as f:
                f.write('Episode: {}/{}, makespan: {}, success: {}/100, eps: {:.2}\nactions: {}\n'.
                        format(e, episode_num, score, succ_count, agent.eps, action_list))
            f.close()
            succ_count = 0

            if type(agent.net) == 'nn.mlp.DropoutMLP':
                torch.save(agent.net.state_dict(), 'saved_model_torch_episode_' + str(e) + '.pt')
            else:
                agent.net.save('saved_model_keras')
        if e % 1000 == 0:
            scenario.plot_scheduling_result(save_path='../../figs/JSSP_scheduling_result_episode_' + str(e) + '.png')


if __name__ == '__main__':
    job_num = 5
    machine_num = 4
    feature_num = 2

    state_size = job_num * machine_num
    action_size = job_num
    replay_size = state_size * 10

    # test JSSP_model
    agent = DQNAgent(state_size, action_size, job_num, feature_num, replay_size)
    train(agent, job_num=5, machine_num=4)

    # test DropoutMLP
    agent = DQNAgent1(state_size, action_size, job_num, feature_num, replay_size)
    train(agent, job_num=5, machine_num=4)
