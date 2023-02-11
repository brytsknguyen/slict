# SLICT: Multi-Input Multi-Scale Surfel-Based Lidar-Inertial Continuous-Time Odometry and Mapping

# Demo Video

<div align="center">
    <a href="https://youtu.be/mogIgBq97Hs" target="_blank">
    <img src="docs/thumbnail.png" width=80% />
</div>

# Prerequisites

The software was developed on the following dependancies
1. [Ubuntu 20.04](https://releases.ubuntu.com/20.04/)
2. [ROS Noetic](http://wiki.ros.org/noetic/Installation)
3. [Ceres 2.1.0](http://ceres-solver.org/installation.html)
4. [UFOMap (devel_surfel)](https://github.com/brytsknguyen/ufomap/tree/devel_surfel)
5. [Livox ROS driver (forked)](https://github.com/brytsknguyen/livox_ros_driver)

Ubuntu 20.04 and ROS Noetic is a must for compiling SLICT due to UFOMap's [minimum requirement](https://github.com/UnknownFreeOccupied/ufomap/wiki/Setup#installation). However a docker can be used to run SLICT with older OS versions. Please find the instructions below.

# Installation
Please install all dependencies first. Afterwards, create a ros workspace, clone the package to the workspace, and build by `catkin build` or `catkin_make`, for e.g.:

```
mkdir catkin_ws/src
cd catkin_ws/src
git clone https://github.com/brytsknguyen/slict
cd ..; catkin build
```
The launch files for NTU VIRAL, Newer College, MCD VIRAL, and FusionPortable are provided under `launch`

Please raise an issue if you encounter any problem.

# Docker User

## Precompiled Image

To use SLICT on any platform without compiling issues, we prepare a [docker image](https://hub.docker.com/repository/docker/brytsknguyen/slict-noetic-focal/general) with all things precomiled.

First, please install docker engine using the instructions at https://docs.docker.com/engine/install .

Once done, pull this repo into your workspace, then build (ignore error), and `source devel/setup.sh` to update the path to packages. For e.g.
    
```
mkdir catkin_ws/src
cd catkin_ws/src
git clone https://github.com/brytsknguyen/slict
catkin build                                        # there may be error
source ../devel/setup.bash                          # now SLICT's path is set
```

Now we can just call the scripts under `docker` for the corresponding dataset. For example you can run SLICT with an NTU VIRAL sequence by

```
roscd slict/docker && ./run_ntuviral.sh /path/to/dataset
```

You can also set a default /path/to/dataset in the script `run_ntuviral.sh` at [this line](https://github.com/brytsknguyen/slict/blob/877cad94b209be94defd6a7c578bd55b349d1024/docker/run_ntuviral.sh#L6).

To change between the sequence, simply change the `bag_file` argument in the [launch file](https://github.com/brytsknguyen/slict/blob/877cad94b209be94defd6a7c578bd55b349d1024/launch/run_ntuviral.launch#L23).

## Recompile SLICT on local container

For advance users who wants to test your changes to SLICT on the local container, simply build one with the provided docker file:

```
roscd slict/docker && make build  # Container is named slict-noetic-focal
```

Afterwards, you can remove the [repository reference](https://github.com/brytsknguyen/slict/blob/877cad94b209be94defd6a7c578bd55b349d1024/docker/run_ntuviral.sh#L7) `brytsknguyen` in the script to use the local container `slict-noetic-focal`


# Publication
The details of SLICT are presented in the [paper](https://arxiv.org/abs/2211.03900) (accepted, pending publication) and further elaborated in the attached slides [slides](https://github.com/brytsknguyen/slict/blob/master/docs/GM_20230126_SLICT.pdf). Please cite this work if you find it useful:

```
@article{nguyen2023slict,
  title     = {SLICT: Multi-input Multi-scale Surfel-Based Lidar-Inertial Continuous-Time Odometry and Mapping},
  author    = {Nguyen, Thien-Minh and Duberg, Daniel and Jensfelt, Patric and Yuan, Shenghai and Xie, Lihua},
  journal   = {IEEE Robotics and Automation Letters},
  volume    = { },
  number    = { },
  pages     = {0--8},
  year      = {2023}
  publisher = {IEEE}
}
```

![image](docs/ba-dum-tsss.gif)
