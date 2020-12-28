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
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, num_agents, lr_actor=1.0e-2, lr_critic=1.0e-2, discount_factor=0.95, tau=0.02):
        super(CooperativeDDPGAgent, self).__init__()

        self.actors = [Actor(in_actor, hidden_in_actor,
                           hidden_out_actor, out_actor).to(device) for _ in range(num_agents)]
        self.critic = Critic(in_critic, hidden_in_critic,
                             hidden_out_critic, 1).to(device)
        self.target_actors = [Actor(
            in_actor, hidden_in_actor, hidden_out_actor, out_actor).to(device) for _ in range(num_agents)]
        self.target_critic = Critic(
            in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)

        self.noise = OUNoise(out_actor, scale=1.0)

        self.num_agents = num_agents

        self.discount_factor = discount_factor
        self.tau = tau

        self.iter = 0

        # Initialize networks
        for actor in self.actors:
            actor.reset_parameters()
        self.critic.reset_parameters()

        # initialize targets same as original networks
        for target_actor, actor in zip(self.target_actors, self.actors):
            hard_update(target_actor, actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizers = [Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=lr_critic, weight_decay=1.e-5)

    
    def _act(self, actors, obs, noise=0.0):
        obs = obs.to(device)
        actions = []
        for agent_num, actor in enumerate(actors):
            indices = torch.tensor([agent_num]).to(device)
            agent_obs = torch.index_select(obs, -2, indices)
            action = actor(agent_obs) + noise*self.noise.noise().to(device)
            actions.append(action)
        return torch.cat(actions, dim=-2)

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
        for agent_num in range(self.num_agents):
            al = self.update_actor(samples, agent_num)

            logger.add_scalars(f'agent{agent_num}/losses',
                                {'actor_loss': al},
                                self.iter)

    def update_actor(self, samples, agent_num):
        obs, *_ = samples

        actor = self.actors[agent_num]
        actor_optimizer = self.actor_optimizers[agent_num]

        # update actor network using policy gradient
        actor_optimizer.zero_grad()

        # Select agent_num's observations
        indices = torch.tensor([agent_num], device=device)
        obs = torch.index_select(obs, 1, indices)

        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = actor(obs)

        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((obs, q_input), dim=2)

        # get the policy gradient
        actor_loss = -self.critic(q_input2).mean()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(),0.5)
        actor_optimizer.step()

        return actor_loss.cpu().detach().item()

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

        target_critic_input = torch.cat((next_obs, target_actions), dim=2)

        with torch.no_grad():
            q_next = self.target_critic(target_critic_input)
            q_next = q_next.squeeze()

        y = reward + self.discount_factor * q_next * (1 - done)
        critic_input = torch.cat((obs, action), dim=2)
        q = self.critic(critic_input)
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
