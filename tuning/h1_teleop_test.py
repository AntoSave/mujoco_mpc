"""This script allows for the teleoperation of the H1 Walk task using the WAD keys."""

import pygame
from datetime import datetime
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import mediapy as media
import numpy as np
import pandas as pd
import pathlib
import scipy
import time
from concurrent.futures import ThreadPoolExecutor
from mujoco_mpc import agent as agent_lib
import scipy.stats

def get_experiment_folder():
    experiment_folder = pathlib.Path(__file__).parent.parent / "experiments" / "h1_teleop_experiments" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not experiment_folder.exists():
        experiment_folder.mkdir(parents=True)
    return experiment_folder 

def log_data_point(experiment_file, data, command, data_cache):
    fw = quat_to_forward_vector(data.qpos[3:7])
    theta = np.arctan2(fw[1], fw[0])
    # Get contact information
    contact_info = { "LEFT": False, "RIGHT": False }
    if np.isin([14,15,16], data.contact.geom2).any():
        contact_info["LEFT"] = True
    if np.isin([29, 30, 31], data.contact.geom2).any():
        contact_info["RIGHT"] = True
    row = {
        'time': data.time,
        'command': command,
        'x': data.qpos[0],
        'y': data.qpos[1],
        'theta': theta,
        'vx': data.qvel[0],
        'vy': data.qvel[1],
        'omega': data.qvel[5],
        'contact_left': contact_info["LEFT"],
        'contact_right': contact_info["RIGHT"]
    }
    data_cache.append(row)
    if len(data_cache) >= 1000:
        df = pd.DataFrame(data_cache)
        df.to_csv(experiment_file, mode='a', header=not experiment_file.exists(), index=False)
        data_cache.clear()

def init_pygame():
    pygame.init()
    pygame.display.set_mode((150, 150))
    pygame.display.set_caption("Hello World")
    
def pygame_handle_events():
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_r]:
        return "RESET"
    elif pressed[pygame.K_w]:
        return "FORWARD"
    elif pressed[pygame.K_a]:
        return "LEFT"
    elif pressed[pygame.K_d]:
        return "RIGHT"
    elif pressed[pygame.K_s]:
        return "BACKWARD"
    return None

def plan_and_get_trajectory(agent):
    for _ in range(1):
        agent.planner_step()
    return agent.best_trajectory()

def quat_to_forward_vector(quat):
    return scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=True).as_matrix()[:2,0]

def vector_to_quat(vector):
    return scipy.spatial.transform.Rotation.from_euler('z', np.arctan2(vector[1], vector[0])).as_quat(canonical=True, scalar_first=True)

def rotate_quat(quat, angle):
    r = scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=True)
    r = r * scipy.spatial.transform.Rotation.from_euler('z', angle)
    return r.as_quat(canonical=True, scalar_first=True)

def get_mocap_reference_1(data, command):
    pos_ref = data.qpos[:3]
    quat_ref = data.qpos[3:7]
    if command == "FORWARD":
        pos_ref = pos_ref + np.array([10, 0, 0])
    elif command == "LEFT":
        pos_ref = pos_ref + np.array([0, 10, 0])
    elif command == "RIGHT":
        pos_ref = pos_ref + np.array([0, -10, 0])
    elif command == "BACKWARD":
        pos_ref = pos_ref + np.array([-10, 0, 0])
    return pos_ref, quat_ref

def get_mocap_reference_2(data, command):
    fw = quat_to_forward_vector(data.qpos[3:7])
    pos_ref = data.qpos[:3]
    quat_ref = data.qpos[3:7]
    if command == "FORWARD":
        pos_ref = pos_ref + np.concatenate([fw, [0]])
    elif command == "LEFT":
        quat_ref = rotate_quat(data.qpos[3:7], np.pi/4)
    elif command == "RIGHT":
        quat_ref = rotate_quat(data.qpos[3:7], -np.pi/4)
    elif command == "BACKWARD":
        pass
    return pos_ref, quat_ref

def get_mocap_reference_3(data, command):
    fw = quat_to_forward_vector(data.qpos[3:7])
    pos_ref = data.qpos[:3]
    quat_ref = data.qpos[3:7]
    if command == "FORWARD":
        pos_ref = pos_ref + np.concatenate([fw, [0]])
    elif command == "LEFT":
        fw = quat_to_forward_vector(rotate_quat(data.qpos[3:7], np.pi/4))
        pos_ref = pos_ref + np.concatenate([fw, [0]])
    elif command == "RIGHT":
        fw = quat_to_forward_vector(rotate_quat(data.qpos[3:7], -np.pi/4))
        pos_ref = pos_ref + np.concatenate([fw, [0]])
    elif command == "BACKWARD":
        pass
    return pos_ref, quat_ref

def get_mocap_reference_4(data, command):
    fw = quat_to_forward_vector(data.qpos[3:7])
    pos_ref = data.qpos[:3]
    fw_mean = data.sensor('torso_forward').data[:2]
    quat_ref = data.qpos[3:7]
    if command == "FORWARD":
        pos_ref = pos_ref + np.concatenate([fw_mean, [0]])
    elif command == "LEFT":
        fw_mean = quat_to_forward_vector(rotate_quat(vector_to_quat(fw_mean), np.pi/4))
        pos_ref = pos_ref + np.concatenate([fw_mean, [0]])
    elif command == "RIGHT":
        fw_mean = quat_to_forward_vector(rotate_quat(vector_to_quat(fw_mean), -np.pi/4))
        pos_ref = pos_ref + np.concatenate([fw_mean, [0]])
    elif command == "BACKWARD":
        pass
    return pos_ref, quat_ref
    

if __name__ == "__main__":
    # Model and agent initialization
    model_path = (
        pathlib.Path(__file__).parent
        / "../build/mjpc/tasks/h1/walk/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    agent = agent_lib.Agent(task_id="H1 Walk", 
                        model=model, 
                        server_binary_path=pathlib.Path(agent_lib.__file__).parent
                        / "mjpc"
                        / "agent_server")
    agent.set_cost_weights({'Posture arms': 0.06, 'Posture torso': 0.05, 'Face goal': 4.0})
    # Video rendering
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_folder = get_experiment_folder()
    experiment_file = experiment_folder / "data.csv"
    video_fps = 60
    video_resolution = (720, 1280)
    frame_count = 0
    video_path = experiment_folder / f"h1_walk_{current_datetime}.mp4"
    if not video_path.parent.exists():
        video_path.parent.mkdir()
    renderer = mujoco.Renderer(model, height=video_resolution[0], width=video_resolution[1])
    # Experiment data logging
    data_cache = []
    TRAJ = []
    log_data_point(experiment_file, data, None, data_cache)
    # Planning settings
    i=0
    steps_per_planning_iteration = 10
    init_pygame()
    with mujoco.viewer.launch_passive(model, data) as viewer, ThreadPoolExecutor() as executor:
        with media.VideoWriter(video_path, fps=video_fps, shape=video_resolution) as video:
            while viewer.is_running():
                # Get command from user
                command = pygame_handle_events()
                if command == "RESET":
                    curr_time = data.time
                    mujoco.mj_resetData(model, data)
                    data.time = curr_time
                data.mocap_pos, data.mocap_quat = get_mocap_reference_3(data, command)
                
                # Planning and control
                agent.set_state(
                    time=data.time,
                    qpos=data.qpos,
                    qvel=data.qvel,
                    act=data.act,
                    mocap_pos=data.mocap_pos,
                    mocap_quat=data.mocap_quat,
                    userdata=data.userdata,
                )
                if i % steps_per_planning_iteration == 0:
                    trajectory = plan_and_get_trajectory(agent)
                data.ctrl = agent.get_action(nominal_action=True)
                mujoco.mj_step(model, data)
                i = i + 1
                
                # Render video
                if frame_count < data.time * video_fps:
                    renderer.update_scene(data, camera="top")
                    pixels = renderer.render()
                    video.add_image(pixels)
                    frame_count += 1
                
                # Synchonize viewer, log data point
                viewer.sync()
                log_data_point(experiment_file, data, command, data_cache)
                if i%10 == 0:
                    TRAJ.append(np.concatenate((np.asarray([data.time]),np.array(data.qpos),np.array(data.qvel))))
            TRAJ = np.stack(TRAJ)
            np.save(experiment_folder / "traj.npy", TRAJ)
            pygame.quit()
            exit()