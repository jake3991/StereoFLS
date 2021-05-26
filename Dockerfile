ARG ROS_DISTRO=kinetic
FROM ros:${ROS_DISTRO}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && apt-get install --no-install-recommends -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p catkin_ws/src && \
    git clone "https://github.com/jake3991/Argonaut.git" "catkin_ws/src/argonaut"

COPY . catkin_ws/src/stereo-sonar

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    rosdep install -iy --from-paths "catkin_ws/src/stereo-sonar" --skip-keys="sonar_oculus" \
    && rm -rf /var/lib/apt/lists/*

RUN /ros_entrypoint.sh catkin_make --directory catkin_ws

RUN sed -i "$(wc -l < /ros_entrypoint.sh)i\\source \"/catkin_ws/devel/setup.bash\"\\" /ros_entrypoint.sh
RUN sed -i "$(wc -l < /ros_entrypoint.sh)i\\roslaunch stereo_sonar stereoSonar.launch &\\" /ros_entrypoint.sh

ENTRYPOINT [ "/ros_entrypoint.sh", "roslaunch", "stereo_sonar", "stereoSonar.launch" ]
