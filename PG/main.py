# 开发者：Bright Fang
# 开发时间：2022/4/12 11:35
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

LearningRate = 0.01
Gamma = 0.9  # Gamma越大越容易收敛
Switch = 0  # 训练、测试切换标志
env = gym.make('CartPole-v1')
env = env.unwrapped
state_number = env.observation_space.shape[0]
action_number = env.action_space.n
'''policygrandient第一步先建网络'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.in_to_y1 = nn.Linear(state_number, 20)
        self.in_to_y1.weight.data.normal_(0, 0.1)
        self.y1_to_y2 = nn.Linear(20, 10)
        self.y1_to_y2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, action_number)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, inputstate):
        inputstate = self.in_to_y1(inputstate)
        inputstate = F.relu(inputstate)
        inputstate = self.y1_to_y2(inputstate)
        inputstate = torch.sigmoid(inputstate)
        act = self.out(inputstate)
        # return act
        return F.softmax(act, dim=-1)


class PG():
    def __init__(self):
        self.policy = Net()
        self.rewards, self.obs, self.acts = [], [], []
        self.renderflag = False
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LearningRate)

    '''第二步 定义选择动作函数'''

    def choose(self, inputstate):
        inputstate = torch.FloatTensor(inputstate)
        probs = self.policy(inputstate).detach().numpy()
        action = np.random.choice(np.arange(action_number), p=probs)
        return action

    '''第三步 存储每一个回合的数据'''

    def store_transtion(self, s, a, r):
        self.obs.append(s)
        self.acts.append(a)
        self.rewards.append(r)

    '''第四步 学习'''

    def learn(self):
        # pass
        discounted_ep_r = np.zeros_like(self.rewards)
        running_add = 0
        for t in reversed(range(0, len(self.rewards))):
            running_add = running_add * Gamma + self.rewards[t]
            discounted_ep_r[t] = running_add  # 例如，discounted_ep_r是1*87的列表，列表的第一个值为58，最后一个值为1
        # 先减去平均数再除以标准差，就可对奖励归一化，奖励列表的中间段为0，最左为+2.1，最右为-1.9.
        discounted_ep_r -= np.mean(discounted_ep_r)
        discounted_ep_r /= np.std(discounted_ep_r)
        discounted_ep_rs_norm = discounted_ep_r
        self.optimizer.zero_grad()
        # 把一个回合的状态、动作、奖励三个列表转为tensor
        self.obs = np.array(self.obs)
        state_tensor = torch.FloatTensor(self.obs)
        reward_tensor = torch.FloatTensor(discounted_ep_rs_norm)
        action_tensor = torch.LongTensor(self.acts)
        # 我们可以用G值直接进行学习，但一般来说，对数据进行归一化处理后，训练效果会更好
        log_prob = torch.log(self.policy(state_tensor))  # log_prob是拥有两个动作概率的张量，一个左动作概率，一个右动作概率
        selected_log_probs = reward_tensor * log_prob[
            np.arange(len(action_tensor)), action_tensor]  # np.arange(len(action_tensor))是log_prob的索引，
        # action_tensor由0、1组成，于是log_prob[np.arange(len(action_tensor)), action_tensor]就可以取到我们已经选择了的动作的概率，是拥有一个动作概率的张量
        loss = -selected_log_probs.mean()
        print(f'loss is {loss}')
        loss.backward()
        self.optimizer.step()
        self.obs, self.acts, self.rewards = [], [], []


'''训练'''
if Switch == 0:
    print("训练PG中...")
    f = PG()
    for i in range(2000):
        r = 0
        observation = env.reset()
        while True:
            if f.renderflag: env.render()
            action = f.choose(observation)
            observation_, reward, done, info = env.step(action)
            # 修改reward
            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r3 = 3 * r1 + r2
            # 你也可以不修改奖励，直接用reward，都能收敛
            f.store_transtion(observation, action, r3)
            r += r3
            if done:
                f.learn()
                break
            observation = observation_
        print("\rEp: {} rewards: {}".format(i, r), end="")
        if i % 10 == 0 and i > 500:
            save_data = {'net': f.policy.state_dict(), 'opt': f.optimizer.state_dict(), 'i': i}
            torch.save(save_data, "model_PG.pth")
else:
    print("测试PG中...")
    c = PG()
    checkpoint = torch.load("model_PG.pth")
    c.policy.load_state_dict(checkpoint['net'])
    for j in range(10):
        state = env.reset()
        total_rewards = 0
        while True:
            env.render()
            state = torch.FloatTensor(state)
            action = c.choose(state)
            new_state, reward, done, info = env.step(action)  # 执行动作
            total_rewards += reward
            if done:
                print("Score", total_rewards)
                break
            state = new_state
    env.close()
