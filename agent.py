import torch

from networks import Actor, Critic
from torch.optim import Adam
from utilities import hard_update, soft_update

# add OU noise for exploration
from OUNoise import OUNoise


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

class CooperativeDDPGAgent:
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, num_agents, lr_actor=1.0e-2, lr_critic=1.0e-2, discount_factor=0.95, tau=0.02):
        super(CooperativeDDPGAgent, self).__init__()

        self.actor = Actor(in_actor, hidden_in_actor,
                           hidden_out_actor, out_actor).to(device)
        self.critic = Critic(in_critic, hidden_in_critic,
                             hidden_out_critic, num_agents).to(device)
        self.target_actor = Actor(
            in_actor, hidden_in_actor, hidden_out_actor, out_actor).to(device)
        self.target_critic = Critic(
            in_critic, hidden_in_critic, hidden_out_critic, num_agents).to(device)

        self.noise = OUNoise(out_actor, scale=1.0)

        self.num_agents = num_agents

        self.discount_factor = discount_factor
        self.tau = tau

        self.iter = 0

        # Initialize networks
        self.actor.reset_parameters()
        self.critic.reset_parameters()

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=lr_critic, weight_decay=1.e-5)

    def act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.actor(obs) + noise*self.noise.noise()
        return action

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.target_actor(obs) + noise*self.noise.noise()
        return action

    def update(self, samples, logger):
        """update the critics and actors of all the agents """

        # First we update the critic
        cl = self.update_critic(samples)

        # Then the actors
        al = self.update_actor(samples)

        logger.add_scalars('agent/losses',
                            {'critic loss': cl,
                            'actor_loss': al},
                            self.iter)

    def update_actor(self, samples):
        obs, *_ = samples

        # update actor network using policy gradient
        self.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = self.actor(obs)

        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((obs, q_input), dim=2)

        # get the policy gradient
        actor_loss = -self.critic(q_input2).mean()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(),0.5)
        self.actor_optimizer.step()

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

        target_critic_input = torch.cat((next_obs, target_actions), dim=2).to(device)

        with torch.no_grad():
            q_next = self.target_critic(target_critic_input)

        y = reward + self.discount_factor * q_next * (1 - done)
        critic_input = torch.cat((obs, action), dim=2).to(device)
        q = self.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        return critic_loss.cpu().detach().item()

    def update_targets(self):
        self.iter += 1
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)
