This repo contains the recently improved/changed code derived from the papers "Fusing Concurrent Orthogonal Wide-aperture Sonar Images for Dense Underwater 3D Reconstruction" (IROS 2020) and "Predictive  3D  Sonar  Mapping  of  Underwater  Environments via  Object-specific  Bayesian  Inference" (ICRA 2021). Note that this codebase is ros native and will require a ros installation. It can be used without ros, but will require some work. 

# Note on python version
I have made the required changes for this repo to run on Python3. I will support any issues related to Python2 but encourage any users to migrate to Python3 and ros-noetic. 

# Dependency Install 
Install pybind11
```
ros-${DISTRO}-pybind11-catkin
```
Presently, this codebase uses our sonar image system in Argonaut
```
git clone https://github.com/jake3991/Argonaut.git
```
You will also need the following packages
```
ros kinetic or better
opencv
python opencv
numpy
scipy
sklearn
```

# Install guide
Set up your catkin workspace and then clone this repo into src
```
git clone <this repo>
```
Compile
```
catkin_make
```

# Usage guide
This system is built around our custom BlueROV2-heavy and its software package, Argonaut (https://github.com/jake3991/Argonaut). The python node subscribes to two image topics and performs sensor fusion, yielding a point cloud. Some important notes. 
  - You may need to change the topics subscribed to, and the message type
  - the rospackage is "stereo_sonar" this is important when using roslauch/rosrun
  - The way we convert images from polar to cartesian is based on our real-world sonar driver. You will likely need to change this, see the function img2Real.
  - I highly recommend reading through the python script stereoSonarCartisian.py, this script is the meat of the code. This script has been commented to stand on its own, marking recommend changes and potential pitfalls. 
  - CFAR is the feature extractor of choice, a CPP implementation is included in this repo and is the original work of Jinkun Wang
  - I highly recommend tuning CFAR as best as you can, garbage in garbage out. Also you may want to consider alternate feature extraction tools. 
  - I have included a launch file to get you started. Follow the above install instuctuions and then call
    - roslaunch stereo_sonar stereoSonar.launch
  - pybind11 has known some issues with python3, some of which can effect your system, you have been warned!
  - This is not the worlds most efficent implmenation (python), it is written to be easy to understand and modify


# ReadMe to dos
  - config file parameters guide
  - some nice gifs
# Other to dos
  - add type hints to documentation

# Citation
If you use this repo in your work, please cite the following papers. 

```
@inproceedings{
  title={Fusing Concurrent Orthogonal Wide-aperture Sonar Images for Dense Underwater 3D Reconstruction},
  author={John McConnell, John D. Martin and Brendan Englot},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020},
  organization={IEEE}
}
@inproceedings{
  title={Predictive  3D  Sonar  Mapping  of  Underwater  Environments via  Object-specific  Bayesian  Inference},
  author={John McConnell and Brendan Englot},
  booktitle={IEEE/RSJ International Conference on Robotics and Automation (ICRA)},
  year={2021},
  organization={IEEE}
}
```
