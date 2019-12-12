import numpy as np
import tensorflow as tf

!wget https://storage.googleapis.com/bandits_datasets/jester_data_40jokes_19181users.npy

def sample_jester_data(file_name, context_dim = 32, num_actions = 8, num_contexts = 19181,shuffle_rows=True, shuffle_cols=False):
    """
    Samples bandit game from (user, joke) dense subset of Jester dataset.
    Args:   
            file_name: Route of file containing the modified Jester dataset.
            context_dim: Context dimension (i.e. vector with some ratings from a user).
            num_actions: Number of actions (number of joke ratings to predict).
            num_contexts: Number of contexts to sample.
            shuffle_rows: If True, rows from original dataset are shuffled.
            shuffle_cols: Whether or not context/action jokes are randomly shuffled.
    Returns:
            dataset: Sampled matrix with rows: (context, rating_1, ..., rating_k).
            opt_vals: Vector of deterministic optimal (reward, action) for each context.
    """

    np.random.seed(0)
    with tf.gfile.Open(file_name, 'rb') as f:
        dataset = np.load(f)
    if shuffle_cols:
        dataset = dataset[:, np.random.permutation(dataset.shape[1])]
    if shuffle_rows:
        np.random.shuffle(dataset)
    dataset = dataset[:num_contexts, :]

    assert context_dim + num_actions == dataset.shape[1], 'Wrong data dimensions.'
    opt_actions = np.argmax(dataset[:, context_dim:], axis=1)
    opt_rewards = np.array([dataset[i, context_dim + a] for i, a in enumerate(opt_actions)])

    return dataset, opt_rewards, opt_actions

ds, opt_rewards, opt_actions = sample_jester_data('jester_data_40jokes_19181users.npy')

print("dataset shape is :", ds.shape)

import matplotlib.pyplot as plt

plt.hist(opt_actions,bins= np.arange(9)-0.5)

"""# UCB Algorithm"""

class LinUCB:
    def __init__(self, num_action=8, num_feature=32, ):
        self.num_action = num_action
        self.num_feature = num_feature
        self.A = [np.identity(self.num_feature) for i in range(self.num_action)]                # 32 * 32
        self.b = [np.zeros((self.num_feature, 1)) for i in range(self.num_action)]               # 32 * 1 
        self.A_inverse = [np.identity(self.num_feature) for i in range(self.num_action)]        # 32* 32
    
    def train(self, context, action, reward):
        '''
        context:  1*32
        reward, action: scalar
        '''
        context = np.matrix(context)
        self.A[action] += context.T * context
        self.b[action] += reward * context.T
        self.A_inverse[action] = np.linalg.solve(self.A[action], np.identity(self.num_feature))
        return(None)

    def predict(self, context):
        context = np.matrix(context)
        rewards = [0 for i in range(self.num_action)]
        for action in range(self.num_action):
            rewards[action] = context.dot(self.A_inverse[action].dot(self.b[action])) + np.sqrt(context.dot( self.A_inverse[action]).dot( context.T))
        
        #print(rewards)
        return(np.random.choice(( np.where(np.array(rewards) == np.max(rewards) ))[0]))

## initalize
model = LinUCB()

##
training_data = ds[:18000]
training_x = training_data[:,:32]
training_r = training_data[:,32:]

## train
for nx,x in enumerate(training_x):
    #n = np.random.choice(( np.where(np.array(training_r[nx]) == max(training_r[nx]) ))[0])
    #n = np.random.choice(8)
    n = model.predict(x)
    model.train(x, n, training_r[nx,n])

    '''
    for n,r in enumerate(training_r[nx]):
        model.train(x, n, r)
    '''

[model.predict(training_x[i]) - opt_actions[i] for i in range(40)]

regret_list=[]
regret = 0
for i in range(18000, ds.shape[0]):
    context = dataset[i, :32]
    rewards = dataset[i, 32:]
    pred_a = model.predict(context)
    regret += opt_rewards[i] - rewards[pred_a]
    regret_list.append(regret)
plt.plot(regret_list)
plt.xlabel('Index')
plt.ylabel('Regrets')
plt.show()

