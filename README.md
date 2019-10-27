# Training-an-agent-to-play-Doom-Game

Doom is a  popular first-person shooter game. The goal of the game is to kill monsters. Doom is another example of a partially observable MDP as the agent's (player) view is limited to 90 degrees. The agent has no idea about the rest of the environment. Now, we will see how can we use DRQN to train our agent to play Doom. 

ViZDoom allows developing AI bots that play Doom using only the visual information (the screen buffer). It is primarily intended for research in machine visual learning, and deep reinforcement learning, in particular.

Setting up the compilation on Windows is really tedious so using the precompiled binaries is recommended.

Vizdoom directory from Python builds contains complete Python package for Windows. You can copy it to your project directory or copy it into python_dir/lib/site-packages/vizdoom to install it globally in your system.




GPU Requirements:

Please make sure to install CUDA 10.0 Toolkit to work with tensorflow-gpu. 
https://developer.nvidia.com/cuda-10.0-download-archive
