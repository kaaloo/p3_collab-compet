from numpy.core.numeric import indices
import torch
import torch.nn.functional as F

from networks import Actor, Critic
from torch.optim import Adam
from utilities import hard_update, soft_update

# add OU noise for exploration
from OUNoise import OUNoise


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CooperativeDDPGAgent:
    def __init__(self, state_size, action_size,
                 hidden_in_actor, hidden_out_actor,
                 hidden_in_critic, hidden_out_critic,
                 num_agents, lr_actor=1.0e-2, lr_critic=1.0e-2, discount_factor=0.95, tau=0.02):
        super(CooperativeDDPGAgent, self).__init__()

        self.actors = [Actor(state_size, hidden_in_actor,
                             hidden_out_actor, action_size).to(device) for _ in range(num_agents)]
        self.critic = Critic(state_size, action_size, hidden_in_critic,
                             hidden_out_critic, 1, num_agents).to(device)
        self.target_actors = [Actor(
            state_size, hidden_in_actor, hidden_out_actor, action_size).to(device) for _ in range(num_agents)]
        self.target_critic = Critic(
            state_size, action_size, hidden_in_critic, hidden_out_critic, 1, num_agents).to(device)

        self.noise = OUNoise(action_size, scale=1.0)

        self.num_agents = num_agents

        self.discount_factor = discount_factor
        self.tau = tau

        self.iter = 0

        # initialize targets same as original networks
        for target_actor, actor in zip(self.target_actors, self.actors):
            hard_update(target_actor, actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizers = [
            Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=lr_critic, weight_decay=1.e-5)

    def _act(self, actors, obs, noise=0.0):
        obs = obs.to(device)
        actions = []
        for agent_num, actor in enumerate(actors):
            indices = torch.tensor([agent_num]).to(device)
            agent_obs = torch.index_select(obs, -2, indices)
            agent_obs = agent_obs.squeeze()
            action = actor(agent_obs) + noise*self.noise.noise().to(device)
            actions.append(action)
        return torch.stack(actions, dim=1)

    def act(self, obs, noise=0.0):
        return self._act(self.actors, obs, noise)

    def target_act(self, obs, noise=0.0):
        return self._act(self.target_actors, obs, noise)

    def update(self, samples, logger):
        """update the critics and actors of all the agents """

        # First we update the critic
        cl = self.update_critic(samples)
        logger.add_scalars('critic/loss', {'critic loss': cl}, self.iter)

        # Then the actors
        als = self.update_actors(samples)

        for agent_num in range(self.num_agents):
            logger.add_scalars(f'agent{agent_num}/losses',
                              {'actor_loss': als[agent_num]},
                              self.iter)

    def update_actors(self, samples):
        obs, *_ = samples

        # update actor network using policy gradient
        for actor_optimizer in self.actor_optimizers:
            actor_optimizer.zero_grad()

        q_input = self.act(obs)

        # get the policy gradient
        actor_loss = -self.critic(obs, q_input).squeeze().mean(dim=0)
        actor_loss.mean().backward()

        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(),0.5)
        for actor_optimizer in self.actor_optimizers:
            actor_optimizer.step()

        return actor_loss.cpu().detach()

    def update_critic(self, samples):
        """update the critics and return loss """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        obs, action, reward, next_obs, done = samples

        self.critic_optimizer.zero_grad()

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)

        with torch.no_grad():
            q_next = self.target_critic(next_obs, target_actions)
            q_next = q_next.squeeze()

        y = reward + self.discount_factor * q_next * (1 - done)
        q = self.critic(obs, action)
        q = q.squeeze()

        critic_loss = F.mse_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        return critic_loss.cpu().detach().item()

    def update_targets(self):
        self.iter += 1
        for target_actor, actor in zip(self.target_actors, self.actors):
            soft_update(target_actor, actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)
