# main function that sets up environments
# perform training loop

import os
import numpy as np
import progressbar as pb
import torch

from buffer import ReplayBuffer
from agent import CooperativeDDPGAgent
from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment
from utilities import make_tensor, transpose_to_tensor


def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)


def pre_process(entity, batchsize):
    processed_entity = []
    for j in range(3):
        list = []
        for i in range(batchsize):
            b = entity[i][j]
            list.append(b)
        c = torch.Tensor(list)
        processed_entity.append(c)
    return processed_entity


def main():
    seeding()
    # number of parallel agents
    # number of training episodes.
    # change this to higher number to experiment. say 30000.
    number_of_episodes = 5000
    episode_length = 300
    batchsize = 1000
    # how many episodes to save policy and gif
    save_interval = 1000

    # amplitude of OU noise
    # this slowly decreases to 0
    noise = 4
    noise_reduction = 0.999999

    discount_factor = 0.95
    tau = 0.02

    # how many episodes before update
    episode_per_update = 2

    log_path = os.getcwd()+"/log"
    model_dir = os.getcwd()+"/model_dir"

    os.makedirs(model_dir, exist_ok=True)

    env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")
    
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents 
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    # keep 5000 episodes worth of replay
    buffer = ReplayBuffer(int(5000*episode_length))

    actor_hidden = state_size*state_size // 2
    critic_hidden = (state_size + action_size) * (state_size + action_size) // 2

    # initialize policy and critic
    agent = CooperativeDDPGAgent(
        state_size,
        action_size,
        actor_hidden, 
        actor_hidden // 2, 
        critic_hidden, 
        critic_hidden // 2, 
        num_agents, 
        discount_factor=discount_factor, 
        tau=tau)
    logger = SummaryWriter(log_dir=log_path)

    all_scores = []

    # training loop
    for episode in range(0, number_of_episodes):

        reward_this_episode = np.zeros(num_agents)
        env_info = env.reset(train_mode=False)[brain_name]
        obs = env_info.vector_observations

        # for calculating rewards for this particular episode - addition of all time steps

        # save info or not
        save_info = ((episode) % save_interval < 1 or episode == number_of_episodes-1)

        for episode_t in range(episode_length):

            # explore = only explore for a certain number of episodes
            # action input needs to be transposed
            actions = agent.act(make_tensor(obs), noise=noise)
            actions = actions.cpu().detach().numpy()
            noise *= noise_reduction

            # step forward one frame
            env_info = env.step(actions)[brain_name]
            next_obs = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            # add data to buffer
            transition = (obs, actions, rewards, next_obs, dones)

            buffer.push(transition)

            reward_this_episode += rewards

            obs = next_obs

        # update once after every episode_per_update
        if len(buffer) > batchsize and episode % episode_per_update < 1:
            samples = buffer.sample(batchsize)
            samples = transpose_to_tensor(samples)
            agent.update(samples, logger)
            agent.update_targets()  # soft update the target network towards the actual networks

        # saving model
        if save_info:
            dicts = [{
                'critic_params': agent.critic.state_dict(),
                'critic_optim_params': agent.critic_optimizer.state_dict()
            }]
            for actor, optimizer in zip(agent.actors, agent.actor_optimizers):
                save_dict = {'actor_params': actor.state_dict(),
                                'actor_optim_params': optimizer.state_dict()}
                dicts.append(save_dict)

            torch.save(dicts,
                        os.path.join(model_dir, 'episode-{}.pt'.format(episode)))


        episode_score = np.max(reward_this_episode)
        all_scores.append(episode_score)

        running_mean = np.mean(all_scores[-min(100, episode):])
        logger.add_scalars('score', {'score': episode_score, 'mean_score': running_mean}, agent.iter)

        if running_mean >= 0.5:
            print(f'Environment solved in {episode} episodes!')
            break

    env.close()
    logger.close()

if __name__ == '__main__':
    main()
