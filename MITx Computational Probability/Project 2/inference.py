#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu) -- updated 10/18/2016
import collections
import sys

import graphics
import numpy as np
import robot
from scipy.special import xlogy

# Throughout the code, we use these variables.
# Do NOT change these (but you'll need to use them!):
# - all_possible_hidden_states: a list of possible hidden states
# - all_possible_observed_states: a list of possible observed states
# - prior_distribution: a distribution over states
# - transition_model: a function that takes a hidden state and returns a
#     Distribution for the next state
# - observation_model: a function that takes a hidden state and returns a
#     Distribution for the observation from that hidden state
all_possible_hidden_states = robot.get_all_hidden_states()
all_possible_observed_states = robot.get_all_observed_states()
prior_distribution = robot.initial_distribution()
transition_model = robot.transition_model
observation_model = robot.observation_model
hidden_dict = {x: i for i, x in enumerate(all_possible_hidden_states)}
obs_dict = {x: i for i, x in enumerate(all_possible_observed_states)}




# You may find this function helpful for computing logs without yielding a
# NumPy warning when taking the log of 0.
def careful_log(x):
    # computes the log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log(x)


# -----------------------------------------------------------------------------
# Functions for you to implement
#

def forward_backward(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """
        
    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #
    num_time_steps = len(observations)
    #    forward_messages = [None] * num_time_steps
    #    forward_messages[0] = prior_distribution
    initial_dist = np.full(440, 1./440)
    transition = np.zeros((440, 440))
    obs_matrix = np.zeros((440, 96))
    forward_messages = np.zeros((num_time_steps, 440))
    

    
    backward_messages = np.zeros((num_time_steps, 440))
    backward_messages[num_time_steps - 1,:] = np.full(440, 1./440)
    
    for i, x in enumerate(all_possible_hidden_states):
        forward_messages[0, i] = prior_distribution[all_possible_hidden_states[i]]
        for k, v in transition_model(x).items():
            transition[i,hidden_dict[k]] = v
        for k, v in observation_model(x).items():
            obs_matrix[i, obs_dict[k]] = v
    global trans_mat, emit_mat, observes
    trans_mat = transition
    emit_mat = obs_matrix
    observes = observations
    emission = []
    for i, obs in enumerate(observations[:-1]):
        
        if obs:
            emission = emit_mat[:,obs_dict[obs]]
        else:
            emission = np.ones(emit_mat[:,1].shape)
            
        forward_messages[i+1, :] = (emission * forward_messages[i]) @ trans_mat
        forward_messages[i+1, :] = forward_messages[i+1, :]/forward_messages[i+1, :].sum()
    for i, obs in enumerate(observations[:0:-1]):
        if obs:
            emission = emit_mat[:,obs_dict[obs]]
        else:
            emission = np.ones(emit_mat[:,1].shape)
        backward_messages[-(i+2), :] = (emission * backward_messages[-(i+1)]) @ trans_mat.T
        backward_messages[-(i+2), :] = backward_messages[-(i+2), :]/backward_messages[-(i+2), :].sum()
            
    global alphas, betas
    alphas = forward_messages
    betas = backward_messages

    dist_list = []
    for i, obs in enumerate(observations):   
        if obs:
            emission = emit_mat[:,obs_dict[obs]]
        else:
            emission = np.ones(emit_mat[:,1].shape)        
        
        marginal = alphas[i] * betas[i] * emission
        marginal = marginal/marginal.sum()
        temp_dist = robot.Distribution()
        for ind, val in enumerate(marginal):
            temp_dist[all_possible_hidden_states[ind]] = val
        dist_list.append(temp_dist)


    return dist_list


def Viterbi(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #
    num_time_steps = len(observations)
    #    forward_messages = [None] * num_time_steps
    #    forward_messages[0] = prior_distribution
    transition = np.zeros((440, 440))
    obs_matrix = np.zeros((440, 96))
    forward_messages = np.zeros((num_time_steps, 440))
    

    
    backward_messages = np.zeros((num_time_steps - 1, 440))
    
    for i, x in enumerate(all_possible_hidden_states):
        forward_messages[0, i] = prior_distribution[all_possible_hidden_states[i]]
        for k, v in transition_model(x).items():
            transition[i,hidden_dict[k]] = v
        for k, v in observation_model(x).items():
            obs_matrix[i, obs_dict[k]] = v
#    trans_mat = xlogy(-1 * np.abs(np.sign(transition)), transition)
#    emit_mat = xlogy(-1 * np.abs(np.sign(obs_matrix)), obs_matrix)
#    observes = observations
#    emission = []
#    forward_messages[0] = xlogy(-1 * np.abs(np.sign(forward_messages[0])), forward_messages[0])
    trans_mat = transition
    emit_mat = obs_matrix

    global observes
    observes = observations
    emission = []
#    trans_mat = -1*np.log(transition)
#    emit_mat = -1 * np.log(obs_matrix)
#    observes = observations
#    emission = []
#    forward_messages[0] = -1 * np.log(forward_messages[0])



    for i, obs in enumerate(observations[:-1]):
        
        if obs:
            emission = emit_mat[:,obs_dict[obs]]
        else:
            emission = np.ones(emit_mat[:,1].shape)
        
        pre_min_calc = (emission * forward_messages[i]) * trans_mat.T
        


        forward_messages[i+1, :] = np.max(pre_min_calc, axis = 1)
        backward_messages[i] = np.argmax(pre_min_calc, axis = 1)
        forward_messages[i+1, :] = forward_messages[i+1, :]/forward_messages[i+1, :].sum()
           
    global map_alphas, map_betas
    map_alphas = forward_messages
    map_betas = backward_messages


    num_time_steps = len(observations)
    estimated_hidden_states = [] # remove this
    
    

    last_value = np.argmax(emit_mat[:,obs_dict[observations[-1]]] * forward_messages[-1])
    estimated_hidden_states.append(all_possible_hidden_states[int(last_value)])

    for tb in backward_messages[::-1]:
#        print(a)
#        a+=1
##        print(last_value)
#        print(int(tb[last_value]))
        estimated_hidden_states.append(all_possible_hidden_states[int(tb[last_value])])
        last_value = tb[last_value]
#    print(observations)
#    print(estimated_hidden_states[::-1])
    global MAP_best
    MAP_best = estimated_hidden_states[::-1]
    
    return estimated_hidden_states[::-1]


def second_best(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #
    num_time_steps = len(observations)
    #    forward_messages = [None] * num_time_steps
    #    forward_messages[0] = prior_distribution
    transition = np.zeros((440, 440))
    obs_matrix = np.zeros((440, 96))
    forward_messages = np.zeros((num_time_steps, 440))
    diffs = np.zeros((num_time_steps - 1, 440))
    global test, test_arg
    test = np.zeros((num_time_steps -1 , 440, 440))
    test_arg = np.zeros((num_time_steps - 1, 440, 440))
    

    
    backward_messages = np.zeros((num_time_steps - 1, 440))
    backward_messages2 = np.zeros((num_time_steps - 1, 440))

    for i, x in enumerate(all_possible_hidden_states):
        forward_messages[0, i] = prior_distribution[all_possible_hidden_states[i]]
        for k, v in transition_model(x).items():
            transition[i,hidden_dict[k]] = v
        for k, v in observation_model(x).items():
            obs_matrix[i, obs_dict[k]] = v
#    trans_mat = xlogy(-1 * np.abs(np.sign(transition)), transition)
#    emit_mat = xlogy(-1 * np.abs(np.sign(obs_matrix)), obs_matrix)
#    observes = observations
#    emission = []
#    forward_messages[0] = xlogy(-1 * np.abs(np.sign(forward_messages[0])), forward_messages[0])
    trans_mat = transition
    emit_mat = obs_matrix

    global observes
    observes = observations
    emission = []
#    trans_mat = -1*np.log(transition)
#    emit_mat = -1 * np.log(obs_matrix)
#    observes = observations
#    emission = []
#    forward_messages[0] = -1 * np.log(forward_messages[0])



    for i, obs in enumerate(observations[:-1]):
        
        if obs:
            emission = emit_mat[:,obs_dict[obs]]
        else:
            emission = np.ones(emit_mat[:,1].shape)
        
        pre_min_calc = (emission * forward_messages[i]) * trans_mat.T
        
        pre_min_calc_sort = -1 * np.sort(-1 * pre_min_calc, axis = 1)
        test[i] = pre_min_calc_sort
        
        arg_min_calc = np.argsort(-1 * pre_min_calc, axis = 1)
        test_arg[i] = arg_min_calc
#        print(pre_min_calc[:,[0,1]])
        forward_messages[i+1, :] = pre_min_calc_sort[:, 0]
        
        backward_messages[i] = arg_min_calc[:, 0]
        diffs[i] = pre_min_calc_sort[:, 0] - pre_min_calc_sort[:, 1]
        
        backward_messages2[i] = arg_min_calc[:, 1]
        forward_messages[i+1, :] = forward_messages[i+1, :]/forward_messages[i+1, :].sum()
           
    global map_alphas2, map_betas2, map_diffs2
    map_alphas2 = forward_messages
    map_betas2 = backward_messages
    map_diffs2 = diffs

    num_time_steps = len(observations)
    estimated_hidden_states = [] # remove this
    
    

    last_value = np.argmax(emit_mat[:,obs_dict[observations[-1]]] * forward_messages[-1])
    estimated_hidden_states.append(all_possible_hidden_states[int(last_value)])
    smallest_diff = 100
    smallest_diff_index = 1000
    i = -1
    
    real_diffs = np.zeros(99)
    
#    for tb in backward_messages[::-1]:
##        print(a)
##        a+=1
###        print(last_value)
#        real_diffs[i] = diffs[i][last_value]
#        print(i, last_value, tb[last_value], diffs[i][tb[last_value]], smallest_diff, smallest_diff_index)
#        if diffs[i][last_value] < smallest_diff:
#            smallest_diff = diffs[i][last_value]
#            smallest_diff_index = i
#        last_value = tb[last_value]
#        i -= 1
#    print(observations)
#    print(np.sort(real_diffs))
#    print(np.argsort(real_diffs))
    start_locs = -1 * np.sort(-1 * emit_mat[:,obs_dict[observations[-1]]] * forward_messages[-1])
    start_args = np.argsort(-1 * emit_mat[:,obs_dict[observations[-1]]] * forward_messages[-1])
    last_value = start_args[0]
    
    estimated_hidden_states.append(all_possible_hidden_states[int(last_value)])
    
    

#    candidates = [(estimated_hidden_states[:], 0)]
    candidates = []
    for i, loc in enumerate(start_locs[1:]):
        if loc > 0:
            candidates.append(([all_possible_hidden_states[int(start_args[i+1])]], -1*np.log(loc) - -1*np.log(start_locs[0])))
#            print('start: ', -1*np.log(loc), -1*np.log(start_locs[0]))
#            print(candidates[-1])
    i = -1
    min_dev = 10000
    complete_cands = []
    for tb in backward_messages[::-1]:
#        print(a)
#        a+=1
##        print(last_value)
#        print(int(tb[last_value]))
        estimated_hidden_states.append(all_possible_hidden_states[int(tb[last_value])])
        for j, cand in enumerate(complete_cands):
            complete_cands[j] = (cand[0] + [all_possible_hidden_states[int(tb[last_value])]], cand[1])
        map_score = -1*np.log(test[i][last_value][0])
#        print(map_score)
            

        next_candidates = [(estimated_hidden_states[:],0)]
#        next_candidates = []
        for cand in candidates:            
            for j, msg in enumerate(list(test[i][hidden_dict[cand[0][-1]]])):  
#                print('msg', msg)
                if msg > 0  and test_arg[i][hidden_dict[cand[0][-1]]][j] != tb[last_value]:
                    next_candidates.append((cand[0] + [all_possible_hidden_states[int(test_arg[i][hidden_dict[cand[0][-1]]][j])]], cand[1] + \
                    (-1*np.log(msg) - map_score)))
                elif msg > 0 and test_arg[i][hidden_dict[cand[0][-1]]][j]  == tb[last_value] and estimated_hidden_states[-2] != cand[0][-1]:
                    complete_cands.append((cand[0] + [all_possible_hidden_states[int(test_arg[i][hidden_dict[cand[0][-1]]][j])]], cand[1] + \
                    (-1*np.log(msg) - map_score)))
#                    print('msg',np.log(msg))


#                    if min_dev > cand[1] + -1*np.log(msg):
#                        min_dev = cand[1] + -1*np.log(msg)
                    
        candidates = next_candidates                            
        last_value = tb[last_value]
        i -= 1


    sorted_list = sorted(complete_cands + candidates[1:], key = lambda x: x[1])
#    print(sorted_list)
    global MAP_second_best
    MAP_second_best = sorted_list[0][0][::-1]
          
#    print(observations)
#    print(estimated_hidden_states[::-1])
    return sorted_list[0][0][::-1][:10]
#    return estimated_hidden_states[::-1]


# -----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from this project's hidden Markov model
    hidden_states = []
    observations = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state = prior_distribution.sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state = hidden_states[-1]
        new_state = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1:  # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


# -----------------------------------------------------------------------------
# Main
#

def main():
    # flags
    make_some_observations_missing = False
    use_graphics = True
    need_to_generate_data = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)
#
#    filename = 'test.txt'
#    hidden_states, observations = robot.load_data(filename)
#    need_to_generate_data = False
#    num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(num_time_steps,
                          make_some_observations_missing)

    print('Running forward-backward...')
    marginals = forward_backward(observations)
    print("\n")

    timestep = 2
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(),
                     key=lambda x: x[1],
                     reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print("\n")

    print('Running Viterbi...')
    estimated_states = Viterbi(observations)
    print("\n")

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states[time_step] is None:
            print('Missing')
        else:
            print(estimated_states[time_step])
    print("\n")

    print('Finding second-best MAP estimate...')
    estimated_states2 = second_best(observations)
    print("\n")

    print("Last 10 hidden states in the second-best MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states2[time_step] is None:
            print('Missing')
        else:
            print(estimated_states2[time_step])
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP estimate and true hidden " +
          "states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states2[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between second-best MAP estimate and " +
          "true hidden states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != estimated_states2[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP and second-best MAP " +
          "estimates:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()


if __name__ == '__main__':
    main()
    

