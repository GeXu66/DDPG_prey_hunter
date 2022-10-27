from ddpg_torch import Agent
import numpy as np
from utils import plotLearning
import os
import simulator
from torch.utils.tensorboard import SummaryWriter

env = simulator.Simulator(2, show_figure=True)
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[6], tau=0.001, env=env,
              batch_size=64, layer1_size=400, layer2_size=300, n_actions=1)

agent.load_models('_episode_2625')
np.random.seed(0)
n_games = 4000
score_history = []
n_steps = 0
tensorboard_file = 'plots/4'
writer = SummaryWriter(tensorboard_file)
if not os.path.exists(tensorboard_file):  # 判断所在目录下是否有该文件名的文件夹
    os.mkdir(tensorboard_file)  # 创建多级目录用mkdirs，单击目录mkdir
for i in range(n_games):
    i += 1
    obs = env.reset()
    done = False
    score = 0
    episode_step = 0
    while not done:
        act = agent.choose_action(obs)
        action = np.append(1, act)
        new_state, reward, done, info = env.step(action)
        agent.remember(obs, act, reward, new_state, int(done))
        # agent.learn()
        score += reward
        obs = new_state
        n_steps += 1
        episode_step += 1
        print('episode_step', episode_step)
        if episode_step > 1500:
            done = True
    # env.render()
    score_history.append(score)

    if i % 25 == 0:
        agent.save_models(name='_episode_' + str(i))
    avg_score = np.mean(score_history[-10:])
    writer.add_scalar('score', score, global_step=i)
    writer.add_scalar('avg_score', avg_score, global_step=i)
    writer.add_scalar('step', n_steps, global_step=i)
    print('episode ', i, 'score %.2f' % score, 'trailing 10 games avg %.3f' % avg_score)

# filename = 'LunarLander-alpha000025-beta00025-400-300.png'
# plotLearning(score_history, filename, window=100)
