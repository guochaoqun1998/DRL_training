import gym
import torch as t
from DQN import DQN
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import imageio
import os


"""
RL cast:
STATE:  Type: Box(2)
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07

Actions:
        Type: Discrete(3)
        Num    Action
        0      Accelerate to the Left
        1      Don't accelerate
        2      Accelerate to the Right

Reward:
         Reward of 0 is awarded if the agent reached the flag (position = 0.5)
         on top of the mountain.
         Reward of -1 is awarded if the position of the agent is less than 0.5.

Starting State:
         The position of the car is assigned a uniform random value in
         [-0.6 , -0.4].
         The starting velocity of the car is always assigned to 0.
"""
"""
DQN with Q-learning procedure:
1. initial the state 
2. choose the action form the DQN network, epsilon-greedy 
3. env.step(action) and obtain the next_state, reward, then we could update the experience pool   until done
4. learning: batch learning (experience replay) choose 32 steps and update the Q NN
5. target policy: update every TARGET_UPDATE_TIME 
6. use render to show the image after many episodes 

"""


class worker():
    def __init__(self,env,learning_rate,discount_rate):
        self.env = env
        ACTION_NUM = self.env.action_space.n  # action number
        VARIANT_NUM_IN_STATE = self.env.observation_space.shape[0]  # number of variant in the state
        self.dqn = DQN(VARIANT_NUM_IN_STATE, ACTION_NUM, learning_rate, discount_rate)

    def work(self,episode_num, episode_max_steps,epsilon,log_path,dqn_path,gif_save=False):
        time_string = time.ctime()
        os.makedirs(time_string)
        log_path = time_string + log_path
        os.makedirs(log_path)
        dqn_path = time_string + dqn_path
        if gif_save !=False:
            gif_path =  time_string + '/gif'
            os.makedirs(gif_path)


        writer = SummaryWriter(log_path)
        success_num=0
        reward_sum_episodes = []
        step_episodes=[]

        for episode in tqdm(range(0, episode_num)):
            current_state = self.env.reset()
            step = 0
            done = False
            reward_sum = 0
            episode_frames =[]
            saveGIF = False
            if gif_save and episode % 50 ==0:
                saveGIF = True
                episode_frames .append( self.env.render(mode='rgb_array') )

            while not done:
                action = self.dqn.choose_action(current_state,epsilon)
                next_state, reward, done, info = self.env.step(action)

                if saveGIF:
                    episode_frames.append(self.env.render(mode='rgb_array'))

                step += 1

            # change the final reward into POSITIVE_REWARD
                if done:
                    reward = POSITIVE_REWARD
                    success_num+=1

                reward_sum += reward
                if step > episode_max_steps:
                    done = True

                self.dqn.save_experience(current_state, action, reward, next_state)
                if self.dqn.start_learning:
                    self.dqn.learning()

                current_state = next_state
            if episode >= 0.98*EPISODE_NUMBER:
                env.render('human')
                time.sleep(0.1)
            writer.add_scalar('reward', reward_sum, episode)
            writer.add_scalar('successful_rate', float(success_num)/float(episode+1), episode)
            reward_sum_episodes.append(reward_sum)
            writer.add_scalar('reward average', float(np.sum(reward_sum_episodes))/float(episode+1), episode)
            writer.add_scalar('step', step, episode)
            step_episodes.append(step)
            writer.add_scalar('step average', float(np.sum(step_episodes))/float(episode+1), episode)

            if saveGIF:
                time_per_step = 0.01
                images = np.array(episode_frames)
                imageio.mimwrite(gif_path+'/episode_{:d}_{:d}_{:d}.gif'.format(int(episode),int(step),int(reward_sum)),images,
                                 subrectangles=True,duration=time_per_step  )
                print('wrote gif')


        self.dqn.save_model(dqn_path)

    def Q_table(self,size = 100, path=False):
        if  path != False:
            self.dqn.eval_net.load_state_dict(t.load(path))
        location_range = np.linspace(-1.2, 0.6, size)
        speed_range = np.linspace(-0.7, 0.7, size)
        Q_table = np.zeros( (size,size),dtype = np.float32 )
        for location in range(0, size):
            for speed in range(0, size):
                state_t = t.FloatTensor([location_range[location], speed_range[speed]])
                Q_table[location][speed] = t.max(self.dqn.eval_net.forward(state_t), -1)[0].data.numpy()
        TICK_SIZE = 5
        speed_range_tick = np.linspace(-0.7, 0.7, TICK_SIZE,endpoint=True)
        location_range_tick = np.linspace(-1.2, 0.6, TICK_SIZE,endpoint=True)
        xlabel = [str(round(speed, 2)) for speed in speed_range_tick]
        ylabel = [str(round(location, 2)) for location in location_range_tick]
        fig, ax = plt.subplots(1, 1)
        surf = ax.imshow(Q_table, cmap=plt.cm.cool, vmin=-200, vmax=200, alpha=None)
        ax.set_xlabel(np.linspace(-0.7, 0.7, TICK_SIZE,endpoint=True))
        ax.set_ylabel(np.linspace(-1.2, 0.6, TICK_SIZE,endpoint=True))
        print( np.linspace(-1.2, 0.6, TICK_SIZE,endpoint=True) )
        print(ylabel)
        ax.set_yticklabels(ylabel)
        ax.set_xticklabels(xlabel)
        ax.set_xlabel('speed')
        ax.set_ylabel('location')
        ax.set_title('Q table')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()



if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    worker = worker(env,0.01,0.95)
    EPISODE_NUMBER = 300 + 1
    POSITIVE_REWARD = 200

    worker.work(EPISODE_NUMBER,2000,0.95,'/log','/MountainCar-v0-dqn-_reward_params_50cells.pkl',True)

