import torch
from collections import deque
from model import Spiking_QNet, QTrainer, LIFNeuron
from settings import *

class Agent:
    """
    Agent class
    agent running and the snake
    """

    def __init__(self, game, pars=None):
        """
        (Agent, Snake, dict()) -> None
        Initialize everything
        get everything that is passed from 
        json file to modify attributes and train model
        """
        if pars is None:
            pars = dict()
        self.n_games = 0
        self.epsilon = pars.get('eps', EPSILON)
        self.eps = pars.get('eps', EPSILON)
        self.gamma = pars.get('gamma', GAMMA) # discount rate
        self.eps_range = pars.get('eps_range', EPS_RANGE)
        print(self.epsilon ,self.eps)
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Spiking_QNet(len(game.get_state()), pars.get('hidden_size', HIDDEN_SIZE), OUTPUT_SIZE)
        self.trainer = QTrainer(self.model, lr=pars.get('lr',LR), gamma=self.gamma)
        self.no_progress_counter = 0  # 增加一个计数器来追踪是否有进展
        self.max_no_progress_steps = 100  # 可以定义一个阈值，超过这个步数就给惩罚
        self.game = game

    def remember(self, *args):
        """
        (Agent, (float, float, float, float, bool)) -> None
        state: current state
        action: current actions
        reward: current immediate rewards
        next_state: get the next state
        done: terminal state point
        append all this attributes to the queue: memory
        do this every frame
        """
        state, action, reward, next_state, done = args

        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """
        (Agent) -> None
        Train after every game is finished using long-term memory
        """
        # get memory
        # if memory is above a certain BATCH SIZE then
        # randomly sample BATCH SIZE memory

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        # get all states, actions, rewards, etc...
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        # Reset the membrane potentials of neurons before training
        for name, module in self.model.named_modules():
            if isinstance(module, LIFNeuron):  # Reset membrane potential for LIF neurons
                module.v = torch.zeros(module.input_size)

        # Train the model using the QTrainer
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, *args):
        """
        (Agent, (float, float, float, float, bool)) -> None
        state: current state
        action: current actions taken by the agent
        reward: current immediate reward
        next_state: next state of the agent
        done: terminal boolean
        train agent every game frame
        """
        state, action, reward, next_state, done = args

        # 重置膜电位（对于 SNN，我们需要在每次训练时重置神经元的状态）
        for name, module in self.model.named_modules():
            if isinstance(module, LIFNeuron):  # 如果是 LIF 神经元，重置膜电位
                module.v = torch.zeros(module.input_size)

        # 调用 QTrainer 中的 train_step 方法进行训练
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        """
        (Agent, float) -> np.array(dtype=int): (1, 3)
        Get an action either from the policy or randomly.
        """
        # Tradeoff exploration / exploitation based on epsilon and eps_range
        self.epsilon = self.eps - self.n_games
        final_move = [0, 0, 0]

        # Check if the move should be random (exploration)
        if is_random_move(self.epsilon, self.eps_range):
            # Randomly select one of the three possible moves (0, 1, or 2)
            move = random.randint(0, 2)
        else:
            # Else, get the best move from the model (exploitation)
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # Now only get the prediction (Q-values)
            move = torch.argmax(prediction).item()  # Choose the action with the highest Q-value

            # 防止 Q 值的索引超出范围，加入检查
            if move not in [0, 1, 2]:
                move = random.randint(0, 2)

        # Update the final_move list with the selected move
        final_move[move] = 1

        return final_move
