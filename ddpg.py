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

    def update(self, samples, logger):
        """update the critics and actors of all the agents """

        # First we update the critic
        cl = self.update_critic(self, samples)

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        obs, obs_full, action, reward, next_obs, next_obs_full, done = samples

        # update actor network using policy gradient
        self.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [self.ddpg_agents[i].actor(ob) if i == agent_number
                   else self.ddpg_agents[i].actor(ob).detach()
                   for i, ob in enumerate(obs)]

        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((obs_full.t(), q_input), dim=1)

        # get the policy gradient
        actor_loss = -self.critic(q_input2).mean()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(),0.5)
        self.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        
        logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)


    def update_critic(self, samples):
        """update the critics and return loss """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        obs, obs_full, action, reward, next_obs, next_obs_full, done = samples

        obs_full = torch.stack(obs_full)
        next_obs_full = torch.stack(next_obs_full)

        self.critic_optimizer.zero_grad()

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)

        target_critic_input = torch.cat(
            (next_obs_full.t(), target_actions), dim=1).to(device)

        with torch.no_grad():
            q_next = self.target_critic(target_critic_input)

        y = reward.view(-1, 1) + self.discount_factor * \
            q_next * (1 - done.view(-1, 1))
        action = torch.cat(action, dim=1)
        critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)
        q = self.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        return critic_loss.cpu().detach().item()


    def update_targets(self):
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)
