# raven-teleop
CSCI5551

# Simulations
 - Gazebo
 - Genesis
 - Mujoco
 - Kineval

# Scripts
## Installation:
- Create mamba/conda environment
```
mamba env create -f env.yaml
```
- Activate the environment using:
```
mamba activate raven_env
```


## Run:
- To test Intel Realsense RGB-D L515 camera:
```
python camera_realsense.py
```

- Run hand Landmark detection:
```
python landmark_detection.py
```

![alt text](assets/landmark_detection.png)

#TODO: Fix YPR-calculation