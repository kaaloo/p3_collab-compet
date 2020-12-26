import torch

from networks import Actor, Critic
from torch.optim import Adam
from utilities import hard_update, soft_update

# add OU noise for exploration
from OUNoise import OUNoise


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, lr_actor=1.0e-2, lr_critic=1.0e-2, tau=0.02):
        super(DDPGAgent, self).__init__()

        self.actor = Actor(in_actor, hidden_in_actor,
                           hidden_out_actor, out_actor).to(device)
        self.critic = Critic(in_critic, hidden_in_critic,
                             hidden_out_critic, 1).to(device)
        self.target_actor = Actor(
            in_actor, hidden_in_actor, hidden_out_actor, out_actor).to(device)
        self.target_critic = Critic(
            in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)

        self.noise = OUNoise(out_actor, scale=1.0)

        self.tau = tau

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

    def update_targets(self):
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)
