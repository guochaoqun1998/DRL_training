import torch as t
import torch.nn as nn
import numpy as np

"""
Parameters used in the DQN
"""
MEMORY_STEP_NUMBER = 1000
MEMORY_LEARNING_NUM = 1000
BATCH_NUMBER = 32    # every time we learning and update the Q NN, we will choose BATCH_NUMBER steps
SUCCESS_BATCH_NUMBER = 5
TARGET_UPDATE_TIME = 200   # every target_update_time, weighting of the target NN will be updated.

class Net(nn.Module):
    """
    two hidden layers
    """
    def __init__(self,variant_num_in_state,action_num):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(variant_num_in_state,50),
            nn.ReLU(),
            nn.Linear(50,action_num),
            #nn.Softmax(dim=-1)   # make the softmax operation in the every row
        )
    def forward(self,state):
        return self.net(state)

    def load_parameters(self,path):
        self.net.load_state_dict( t.load(path) )

class DQN():

    """
    establish
    1. the experience pool: current_state, action, reward, next_state
    2. experience pool update
    """
    def __init__(self,variant_num_in_state,action_num,LEARNING_RATE,DISCOUNT_RATE):

        self.variant_num = variant_num_in_state
        self.action_num = action_num

        """ build the Q NN"""
        self.eval_net = Net(variant_num_in_state,action_num)
        self.target_net = Net(variant_num_in_state,action_num)

        self.optimizer = t.optim.Adam(self.eval_net.parameters(),lr=LEARNING_RATE)
        # the learning rate
        # use the Adam optimizer

        self.loss_function = nn.MSELoss()
        # loss function

        self.memory_experience = np.zeros( (MEMORY_STEP_NUMBER, variant_num_in_state*2+1+1) ,dtype = np.float)
        self.memory_count = 0
        self.memory_successful_experience = []
        self.memory_success_count = 0
        # memory experience

        self.target_update_count = 0   # judge if we could update the target NN

        self.gamma = DISCOUNT_RATE
        self.learning_rate = LEARNING_RATE

        self.start_learning = False

    def choose_action(self,state,epsilon):
        # epsilon greedy policy
        if np.random.uniform() < epsilon:
            state = t.FloatTensor(state)
            action = t.max( self.eval_net.forward(state),-1 )[1].data.numpy()
        else:
            action = np.random.randint(0,self.action_num)
        return action

    def save_experience(self,current_state,action,reward,next_state):

        experience_list = np.hstack( (current_state,[action,reward],next_state) )
        self.memory_count += 1
        if self.memory_count > MEMORY_STEP_NUMBER:
            self.start_learning = True

        index = self.memory_count % MEMORY_STEP_NUMBER
        self.memory_experience[index,:] = experience_list  # update the experience

    def learning(self):
        if self.target_update_count % TARGET_UPDATE_TIME == 0:
            self.target_net.load_state_dict( self.eval_net.state_dict() )
        self.target_update_count += 1

        learning_index_set = np.random.choice(range(0,MEMORY_STEP_NUMBER),BATCH_NUMBER)
        learning_set = self.memory_experience[learning_index_set,:]

        current_state_set_t = t.FloatTensor( learning_set[:,0:self.variant_num] )
        action_set_t = t.LongTensor( learning_set[:,self.variant_num:self.variant_num+1] )
        reward_set_t = t.FloatTensor( learning_set[:,self.variant_num+1:self.variant_num+2] )
        next_state_set_t = t.FloatTensor( learning_set[:, -self.variant_num:] )

        q_eval = self.eval_net.forward(current_state_set_t).gather(1,action_set_t)

        target_all_value = self.target_net.forward(next_state_set_t).detach()

        q_target = self.gamma * target_all_value.max(1)[0].view(BATCH_NUMBER,1) + reward_set_t

        loss = self.loss_function(q_eval,q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self,file_path1):
        print('model saved')
        t.save(self.eval_net.state_dict(),file_path1)


if __name__ == '__main__':

    net=Net(5,2)
    state = [1.,2.,3.,4.,5.]
    state = t.FloatTensor(state)
    print(net.forward(state))

