""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import _pickle as pickle
import gym

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
if resume:
    model = pickle.load(open('save-3a.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
    model['W2'] = np.random.randn(3,H) / np.sqrt(H) # Kendrick: 3 actions

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up
# gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h<0] = 0 # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    logp -= np.max(logp) # stabilize f

    #p = sigmoid(logp)
    f = np.exp(logp)
    fsum = np.sum(f)
    p = f / fsum

    cache = (x, h, f, fsum)
    return p, cache # return probabilities of taking each 3 action, and cache

def policy_backward(epdp, cache):
    """ backward pass. (eph is array of intermediate hidden states) """
    (epx, eph, epf, epfsum) = cache

    epdfsum = -1.0 / epfsum ** 2 * np.sum(epf * epdp, axis=1).reshape(-1, 1) # E,1
    epdf = 1.0 / epfsum * epdp + np.ones_like(epf) * epdfsum # E,3
    epdlogp = epf * epdf # E,3
    dW2 = np.dot(epdlogp.T, eph) # 3,E x E,H
    epdh = np.dot(epdlogp, model['W2']) # E,3 x 3,H = E,H
    epdh[eph <= 0] = 0 # backpro prelu
    dW1 = np.dot(epdh.T, epx) # H,E x E,D = H,D

    return {'W1':dW1, 'W2':dW2}

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,fs,fsums,dps,drs = [],[],[],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
while True:
    if render: env.render()

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob, cache = policy_forward(x)
    x, h, f, fsum = cache
    dice = np.random.uniform()
    if dice < aprob[0]:
        y = 0
        action = 0 # noting
    elif dice > aprob[0] + aprob[1]:
        y = 2
        action = 2 # up
    else:
        y = 1
        action = 5 # down

    # record various intermediates (needed later for backprop)
    xs.append(x) # observation
    hs.append(h) # hidden state
    fs.append(f)
    fsums.append(fsum)
    onehot = np.eye(3)[y]
    #dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    dps.append(- 1.0 / aprob * onehot) # kendrick: the sign was reversed!

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

    if done: # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epf = np.vstack(fs)
        epfsum = np.vstack(fsums)
        epdp = np.vstack(dps)
        epr = np.vstack(drs)
        xs,hs,fs,fsums,dps,drs = [],[],[],[],[],[] # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        #epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        epdp = epdp * discounted_epr # kendrick: 3 actions
        cache = (epx, eph, epf, epfsum)
        grad = policy_backward(epdp, cache)
        #for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch
        for k in model: grad_buffer[k] += -grad[k] # kendrick: the sign was reversed!

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k,v in model.items():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' %
              (reward_sum, running_reward))
        if episode_number % 100 == 0: pickle.dump(model, open('save-3a.p', 'wb'))
        reward_sum = 0
        observation = env.reset() # reset env
        prev_x = None

    if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
            print('ep %d: game finished, reward: %f' % (episode_number, reward) + '' if reward == -1 else ' !!!!!!!!')
