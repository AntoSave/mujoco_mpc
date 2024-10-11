"""This script is used to test the MujocoMPC baseline agent in a given envorionment with obstacles.
The expected behavior is that the agent will collide with the obstacles as the planning horizon of
MJPC is too short to avoid them."""

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

def quat_to_forward_vector(quat):
    return scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=True).as_matrix()[:2,0]

def vector_to_quat(vector):
    return scipy.spatial.transform.Rotation.from_euler('z', np.arctan2(vector[1], vector[0])).as_quat(canonical=True, scalar_first=True)

def rotate_quat(quat, angle):
    r = scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=True)
    r = r * scipy.spatial.transform.Rotation.from_euler('z', angle)
    return r.as_quat(canonical=True, scalar_first=True)
    

AGENTS = {
    "baseline_1": {
        "task_id": "H1 Walk",
        "cost_weights": {'Posture arms': 0.06, 'Posture torso': 0.05, 'Face goal': 4.0}
    },
    "baseline_2": {
        "task_id": "H1 WalkObs",
        "cost_weights": {'Obstacles': 10.0, "Control": 0.0001}
    },
    "baseline_2_no_obs": {
        "task_id": "H1 WalkObs",
        "cost_weights": {'Obstacles': 0.0, "Control": 0.0001}
    }
}

ENVIRONMENTS = {
    "cylinders": {
        "obstacles": {
            "cylinders": np.array([[3.0, 3.0], [5.0, 5.0]]),
            "sliding_walls": np.array([])
        },
        "obstacles_data": {
            "sliding_walls": np.array([]),
            "trigger": False,
            "transition_end": False
        }, 
        "trigger_cond": lambda data: False,
        "start_pos": np.array([0.0, 0.0]),
        "start_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "goal_pos": np.array([10.0, 10.0]),
    },
    "sliding_walls": {
        "obstacles": {
            "cylinders": np.array([]),
            "sliding_walls": np.array([[[3.33,0.0], [3.33,3.33], [0.0,3.33]],
                                   [[6.66,0.0], [6.66,3.33], [0.0,3.33]],
                                   [[0.0, 6.66], [3.33, 6.66], [3.33, 0.0]]])
        },
        "obstacles_data": {
            "sliding_walls": np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            "trigger": False,
            "transition_end": False
        },
        "trigger_cond": lambda data: data.qpos[1] > 2.5,
        "start_pos": np.array([5.0, 0.0]),
        "start_quat": rotate_quat(np.array([1.0, 0.0, 0.0, 0.0]), np.pi/2),
        "goal_pos": np.array([6.0, 10.0]),
    }
}

def get_experiment_folder():
    experiment_folder = pathlib.Path(__file__).parent.parent / "experiments" / "h1_baseline" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

def plan_and_get_trajectory(agent):
    for _ in range(1):
        agent.planner_step()
    return agent.best_trajectory()

def add_obstacles(model_spec, obstacles):
    for i, o in enumerate(obstacles['cylinders']):
        body = model_spec.worldbody.add_body()
        body.name = f"obstacle_{i}"
        body.pos = o.tolist() + [0.5]
        geom = body.add_geom()
        geom.name = f"obstacle_geom_{i}"
        geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        geom.size = [0.8, 0.8, 0.5]
        geom.rgba = [1, 0, 0, 1]
        
    for i, o in enumerate(obstacles['sliding_walls']):
        start_point = o[0]
        end_point = o[1]
        mean_point = (start_point + end_point)/2
        yaw = np.arctan2(end_point[1]-start_point[1], end_point[0]-start_point[0])
        length = np.linalg.norm(end_point-start_point)
        body = model_spec.worldbody.add_body()
        body.name = f"obstacle_swall_{i}"
        body.pos = mean_point.tolist() + [1.0]
        body.quat = scipy.spatial.transform.Rotation.from_euler('z', yaw).as_quat(scalar_first=True)
        geom = body.add_geom()
        geom.name = f"obstacle_swall_geom_{i}"
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [length/2, 0.2, 1.0]
        geom.rgba = [1, 0, 0, 1]
        
def update_dynamic_obs(obstacles, obstacles_data, mujoco_model):
    if not obstacles_data['trigger']:
        return
    transition_end = True
    for (i, (o, o_pos)) in enumerate(zip(obstacles['sliding_walls'], obstacles_data['sliding_walls'])):
        translation = np.linalg.norm(o[2])
        direction = o[2]/translation
        if np.dot(direction, o_pos) >= translation:
            continue
        transition_end = False
        o_pos += direction * mujoco_model.opt.timestep * 10.0
        start_point = o[0]
        end_point = o[1]
        mean_point = (start_point + end_point)/2
        new_mean_point = mean_point + o_pos
        body = mujoco_model.body(f"obstacle_swall_{i}")
        body.pos = new_mean_point.tolist() + [1.0]
        #body.xipos = body.xpos
    obstacles_data['transition_end'] = transition_end

if __name__ == "__main__":
    selected_agent = "baseline_2_no_obs"
    selected_environment = "sliding_walls"
    obstacles = ENVIRONMENTS[selected_environment]["obstacles"]
    # Model and agent initialization
    model_path = (
        pathlib.Path(__file__).parent
        / "../build/mjpc/tasks/h1/walk/task.xml"
    )
    model_spec = mujoco.MjSpec()
    model_spec.from_file(str(model_path))
    add_obstacles(model_spec, obstacles)
    print(model_spec)
    print(dir(model_spec))
    print(model_spec.meshdir)
    print(model_spec.mesh)
    input("Press Enter to continue...")
    model = model_spec.compile()
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    data.qpos[:2] = ENVIRONMENTS[selected_environment]["start_pos"]
    data.qpos[3:7] = ENVIRONMENTS[selected_environment]["start_quat"]
    data.mocap_pos[0,:2] = ENVIRONMENTS[selected_environment]["goal_pos"]
    data.mocap_quat[0] = rotate_quat(np.array([0, 0, 0, 1]), np.pi/2)
    agent = agent_lib.Agent(task_id=AGENTS[selected_agent]["task_id"], 
                        model=model, 
                        server_binary_path=pathlib.Path(agent_lib.__file__).parent
                        / "mjpc"
                        / "agent_server")
    agent.set_cost_weights(AGENTS[selected_agent]["cost_weights"])
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
    # TRAJ = []
    log_data_point(experiment_file, data, None, data_cache)
    # Planning settings
    i=0
    steps_per_planning_iteration = 1
    done = False
    with mujoco.viewer.launch_passive(model, data) as viewer, ThreadPoolExecutor() as executor:
        with media.VideoWriter(video_path, fps=video_fps, shape=video_resolution) as video:
            while viewer.is_running():
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
                
                # Update dynamic obstacles
                if ENVIRONMENTS[selected_environment]["trigger_cond"](data) and not ENVIRONMENTS[selected_environment]["obstacles_data"]["trigger"]:
                    ENVIRONMENTS[selected_environment]["obstacles_data"]['trigger'] = True
                update_dynamic_obs(ENVIRONMENTS[selected_environment]["obstacles"], ENVIRONMENTS[selected_environment]["obstacles_data"], model)
                
                log_data_point(experiment_file, data, None, data_cache)
            #     if i%10 == 0:
            #         TRAJ.append(np.concatenate((np.asarray([data.time]),np.array(data.qpos),np.array(data.qvel))))
            # TRAJ = np.stack(TRAJ)
            # np.save(experiment_folder / "traj.npy", TRAJ)
            #pygame.quit()
            exit()