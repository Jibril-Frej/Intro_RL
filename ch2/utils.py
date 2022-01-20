import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def e_greedy_bandit(k, max_steps, eps, stationnary=True, alpha=None, init=0):
    # Possible actions
    a = np.arange(0, k, 1, dtype=int)

    # Action values
    q = np.random.normal(0, 1, k)

    # Action values estimates
    Q = np.full(k, init, dtype=float)

    # Count of action chosen
    N = np.zeros(k)

    rewards = np.zeros(max_steps)
    for i in range(max_steps):
        p = np.random.uniform()

        if p < eps:
            A = np.random.choice(a)
        else:
            A = np.random.choice(np.argwhere(Q == Q.max()).flatten())

        # Reward is selected from a normal dist with mean = action value and
        # variance = 1
        R = np.random.normal(q[A], 1)

        N[A] += 1
        if alpha:
            Q[A] += alpha*(R - Q[A])
        else:
            Q[A] += 1/N[A]*(R - Q[A])
        rewards[i] = R
        if not stationnary:
            q += np.random.normal(0, 0.01, k)

    return rewards


def UCB_bandit(k, max_steps, c, stationnary=True, alpha=None):
    # Action values
    q = np.random.normal(0, 1, k)

    # Action values estimates
    Q = np.zeros(k)

    # Count of action chosen
    N = np.zeros(k)

    rewards = np.zeros(max_steps)
    for i in range(max_steps):
        if 0 in N:
            A = np.random.choice(np.argwhere(N == 0).flatten())
        else:
            A = np.argmax(Q + c*np.sqrt(np.log(i+1)/N))

        # Reward is selected from a normal dist with mean = action value and
        # variance = 1
        R = np.random.normal(q[A], 1)

        N[A] += 1
        if alpha:
            Q[A] += alpha*(R - Q[A])
        else:
            Q[A] += 1/N[A]*(R - Q[A])
        rewards[i] = R
        if not stationnary:
            q += np.random.normal(0, 0.01, k)

    return rewards


def gradient_bandit(k, max_steps, alpha, stationnary=True):
    # Possible actions
    a = np.arange(0, k, 1, dtype=int)

    # Action values
    q = np.random.normal(0, 1, k)

    # Action values estimates
    H = np.zeros(k)

    baseline = 0

    rewards = np.zeros(max_steps)
    for i in range(max_steps):
        pi = softmax(H)
        A = np.random.choice(a, p=pi)

        # Reward is selected from a normal dist with mean = action value and
        # variance = 1
        R = np.random.normal(q[A], 1)

        baseline += 1/(i+1)*(R - baseline)
        rewards[i] = R

        for action in a:
            H[action] += alpha*(R-baseline)*(int(action == A)-pi[action])

        if not stationnary:
            q += np.random.normal(0, 0.01, k)

    return rewards
