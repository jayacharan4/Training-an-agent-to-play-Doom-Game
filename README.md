# Training-an-agent-to-play-Doom-Game

Doom is a  popular first-person shooter game. The goal of the game is to kill monsters. Doom is another example of a partially observable MDP as the agent's (player) view is limited to 90 degrees. The agent has no idea about the rest of the environment. Now, we will see how can we use DRQN to train our agent to play Doom. 

ViZDoom allows developing AI bots that play Doom using only the visual information (the screen buffer). It is primarily intended for research in machine visual learning, and deep reinforcement learning, in particular.

Setting up the compilation on Windows is really tedious so using the precompiled binaries is recommended.

vizdoom directory from Python builds contains complete Python package for Windows. You can copy it to your project directory or copy it into python_dir/lib/site-packages/vizdoom to install it globally in your system.

Run CMake GUI, select ViZDoom root directory and set paths to:

BOOST_ROOT
BOOST_INCLUDEDIR
BOOST_LIBRARYDIR
PYTHON_INCLUDE_DIR (optional, for Python/Anaconda bindings)
PYTHON_LIBRARY (optional, for Python/Anaconda bindings)
NUMPY_INCLUDES (optional, for Python/Anaconda bindings)
LUA_LIBRARIES (optional, for Lua/Torch bindings)
LUA_INCLUDE_DIR (optional, for Lua/Torch bindings)
ZDoom dependencies paths
In configuration select BUILD_PYTHON, BUILD_PYTHON3 and BUILD_JAVA options for Python and Java bindings (optional, default OFF).

Use generated Visual Studio solution to build all parts of ViZDoom environment.

Compilation output
Compilation output will be placed in vizdoom_root_dir/bin and it should contain following files.

bin/vizdoom / vizdoom.exe - ViZDoom executable
bin/vizdoom.pk3 - resources file used by ViZDoom (needed by ViZDoom executable)
bin/libvizdoom.a / vizdoom.lib - C++ ViZDoom static library
bin/libvizdoom.so / vizdoom.dll / libvizdoom.dylib - C++ ViZDoom dynamically linked library
bin/python2/vizdoom.so / vizdoom.pyd / vizdoom.dylib - ViZDoom Python 2 module
bin/pythonX.X/vizdoom.so / vizdoom.pyd - ViZDoom Python X.X module
bin/pythonX.X/pip_package - complete ViZDoom Python X.X package
bin/lua/vizdoom.so / vizdoom.so / vizdoom.dylib - ViZDoom Lua C module
bin/lua/luarocks_package - complete ViZDoom Torch package
bin/java/libvizdoom.so / vizdoom.dll / libvizdoom.dylib - ViZDoom library for Java
bin/java/vizdoom.jar - Contains ViZDoom Java classes
Manual installation
To manually install Python package copy vizdoom_root_dir/bin/pythonX.X/pip_package contents to python_root_dir/lib/pythonX.X/site-packages/site-packages/vizdoom.



GPU Requirements:

Please make sure to install CUDA 10.0 Toolkit to work with tensorflow-gpu. 
https://developer.nvidia.com/cuda-10.0-download-archive
