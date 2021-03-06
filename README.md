[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

## Python Dependencies

The following instructions are based on having Miniconda installed on your system.  If that is not the case, please follow the [Miniconda installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

1. Create a conda python 3.6 environment using:

    ```conda create -n p3 python=3.6```

2. Activate the new environement using:

    ```conda activate p3```

3. Install the Udacity DRLND python dependencies by running:

    ```pip install ./python```

4. Install project specific dependencies by running:
    ```pip install -r requirements.txt```


## Submission

The submission consists of this README file explaining how to setup project dependencies, the REPORT file, the project source files and the tensorboard logs and saved models.


The following table describes the different source files in this submission.
File | Description
---------|----------
 agent.py | Collaborative MADDPG agent implementation
 buffer.py | Replay buffer
 main.py | Main (training) script
 networks.py | Neural networks for Actor and Critic
 OUNoise.py | Ornstein-Uhlenbeck noise generator
 utilities.py | Various utility functions

The `log` directory contains the tensorboard log files which can be viewed by running the following command and opening a browser window to https://localhost:3000

```bash
tensorboard --logdir=./log/ --host=0.0.0.0 --port=3000 &> /dev/null
```

The `model_dir` directory contains models saved during training and in particular the `model_dir/episode-1748.pt` file which is the model that solved the Tennis environement.

The following command then runs the training script.  The `log` and `model_dir` directories need to be removed to avoid accumulating data with previous runs.

```bash
python main.py
```