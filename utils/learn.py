import numpy as np
from .utils import to_variable
import random
import torch

def egreedy_action(Q, phi_batch, step, decay_steps=1000.00, greedy=False,
        min_eps=0.05):
    '''
    @phi_batch: Each element of the list is a state-action feature vector
    @step: step in the RL algorithm, used for decaying epsilon.
    '''
    # Initial values
    initial_epsilon, final_epsilon = 1.0, min_eps
    # Calculate step size to move from final to initial epsilon with #decay_steps
    step_size = (initial_epsilon - final_epsilon) / decay_steps
    # Calculate annealed epsilon
    ann_eps = initial_epsilon - step * step_size
    # Set epsilon as max(min_eps, annealed_epsilon)
    epsilon = max(min_eps, ann_eps)

    # for debugging, we always calculate the optimal Qvalue, and return it,
    # even if we are using epsilon.

    phi_batch = to_variable(phi_batch).float()
    all_rewards = Q(phi_batch)
    _, best_action = all_rewards.max(0)
    all_qvals = all_rewards.data.cpu().numpy().flatten()
    # FIXME: why did we need [0][0] here before?
    # best_action = int(best_action.data.cpu().numpy()[0][0])
    best_action = int(best_action.data.cpu().numpy()[0])

    if np.random.uniform() < epsilon and not greedy:
        best_action = random.choice(range(len(phi_batch)))

    return best_action, all_qvals, epsilon

def Qvalues(phi_batch, Q):
    '''
    Note: for each state_i \in state_mb, there is only one action in
    actions_mb.
    returns an array with the Qvalues achieved after passing Q(state_i concat
    actions_i).
    '''
    phi_batch = to_variable(np.array(phi_batch)).float()
    all_vals = Q(phi_batch)
    return all_vals

def Qtargets(r_mb, new_state_action_mb, done_mb, Q_, gamma=1.0):
    '''
    '''
    maxQ = []
    for i, phi_batch in enumerate(new_state_action_mb):
        done = done_mb[i]
        assert done == 0 or done == 1, "sanity check"
        phi_batch = to_variable(np.array(phi_batch)).float()
        if len(phi_batch) == 0:
            assert done, "should be 0 only when done"
            # last reward is always 0
            maxQ.append(0.00)
            continue
        all_vals = Q_(phi_batch)
        # best_reward = all_vals.max().data[0]
        best_reward = all_vals.max().item()
        maxQ.append(best_reward)

    assert len(maxQ) == len(r_mb)
    # now we can find the target qvals using standard reward +
    # q(new_state)*discount_factor formula.
    # Note: we don't care about 'done' because we already take care of
    # situations where episode was done based on the length of the actions
    # array.
    maxQ = to_variable(np.array(maxQ)).float()
    r_mb = to_variable(np.array(r_mb)).float()
    target = r_mb + gamma*maxQ
    return target

def gradient_descent(y, q, optimizer):
    assert y.shape == q.shape, "must match!"
    # Clear previous gradients before backward pass
    optimizer.zero_grad()

    # Run backward pass
    error = (y - q)

    # PN: is this important? Should probably depend on the range we choose for rewards?
    # Clip error to range [-1, 1]
    # error = error.clamp(min=-1, max=1)

    # Square error
    error = error**2
    error = error.sum()

    # q.backward(error.data)
    error.backward()

    # Perfom the update
    optimizer.step()

    return error



