"""Test script for the obstacle avoidance task with absolute motion services"""

from datetime import datetime
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import mediapy as media
import numpy as np
import pathlib
import scipy
import time
from concurrent.futures import ThreadPoolExecutor
from mujoco_mpc import agent as agent_lib
import scipy.stats

model_path = (
        pathlib.Path(__file__).parent
        / "../build/mjpc/tasks/h1/walk/task.xml"
    )
model = mujoco.MjModel.from_xml_path(str(model_path))
model.opt.timestep = 0.002
# data
data = mujoco.MjData(model)
# agents
agent_x = agent_lib.Agent(task_id="H1 Walk", 
                        model=model, 
                        server_binary_path=pathlib.Path(agent_lib.__file__).parent
                        / "mjpc"
                        / "agent_server")
agent_y = agent_lib.Agent(task_id="H1 Walk",
                          model=model,
                            server_binary_path=pathlib.Path(agent_lib.__file__).parent
                            / "mjpc"
                            / "agent_server")
# agent_x.set_cost_weights({'Face goal':0.0, 'Posture up': 0.1})
# agent_y.set_cost_weights({'Face goal':0.0, 'Posture up': 0.1})
agent_x.set_cost_weights({'Posture up': 0.1})
agent_y.set_cost_weights({'Posture up': 0.1})

# Experiment info
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_name = f"h1_walk_{current_datetime}"
experiment_folder = pathlib.Path(__file__).parent.parent / "experiments" / "absolute_policy_combination" / experiment_name
if not experiment_folder.exists():
    experiment_folder.mkdir(parents=True)

# Video
video_fps = 60
video_resolution = (720, 1280)
frame_count = 0

video_path = experiment_folder / f"h1_walk_{current_datetime}.mp4"
if not video_path.parent.exists():
    video_path.parent.mkdir()
renderer = mujoco.Renderer(model, height=video_resolution[0], width=video_resolution[1])

current_agent = 0
steps_per_planning_iteration = 10
i = 0
goal = np.array([10.0, 10.0])
obstacle = np.array([3.0, 3.0])

def plan_and_get_trajectory(agent):
    for _ in range(10):
        agent.planner_step()
    return agent.best_trajectory()

def quat_to_forward_vector(quat):
    return scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=True).as_matrix()[:2,0]

def rotate_quat(quat, angle):
    r = scipy.spatial.transform.Rotation.from_quat(quat, scalar_first=True)
    r = r * scipy.spatial.transform.Rotation.from_euler('z', angle)
    return r.as_quat(canonical=True, scalar_first=True)

def cost(x):
    """Cost function for the planner. x is a two dimensional state vector."""
    return np.linalg.norm(x - goal) + 30*scipy.stats.multivariate_normal.pdf(x, mean=obstacle, cov=0.4*np.eye(2))

X, Y = np.meshgrid(np.linspace(-1, 11, 100), np.linspace(-1, 11, 100))
z = np.array([cost(np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = z.reshape(X.shape)
ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()
input("Press Enter to continue...")

TRAJ = []

with mujoco.viewer.launch_passive(model, data) as viewer, ThreadPoolExecutor() as executor:
    with media.VideoWriter(video_path, fps=video_fps, shape=video_resolution) as video:
        while viewer.is_running():
            
            # set planner state
            agent_x.set_state(
                time=data.time,
                qpos=data.qpos,
                qvel=data.qvel,
                act=data.act,
                mocap_pos=data.qpos[:3] + np.asarray([10.0, 0.0, 0.0]),
                mocap_quat=data.mocap_quat,
                userdata=data.userdata,
            )
            agent_y.set_state(
                time=data.time,
                qpos=data.qpos,
                qvel=data.qvel,
                act=data.act,
                mocap_pos= data.qpos[:3] + np.asarray([0.0, 10.0, 0.0]),
                mocap_quat=data.mocap_quat,
                userdata=data.userdata,
            )
            if i % steps_per_planning_iteration == 0:
                t = time.time()
                f1 = executor.submit(plan_and_get_trajectory, agent_x)
                f2 = executor.submit(plan_and_get_trajectory, agent_y)
                traj1 = f1.result()
                traj2 = f2.result()
                elapsed = time.time() - t
                print("Elapsed planning time: ", elapsed)
                traj1_states = traj1["states"]
                traj2_states = traj2["states"]
                #euler distance from goal
                traj1_goal_distance = cost(traj1_states[-1,:2]) #cost(data.qpos[:2]+np.asarray([0.5, 0.0])) 
                traj2_goal_distance = cost(traj2_states[-1,:2]) #cost(data.qpos[:2]+np.asarray([0.0, 0.5]))
                current_agent = 0 if traj1_goal_distance < traj2_goal_distance else 1

            if current_agent == 0:
                agent = agent_x
            else:
                agent = agent_y
            data.ctrl = agent.get_action(nominal_action=True)
            mujoco.mj_step(model, data)
            v = quat_to_forward_vector(data.qpos[3:7])
            v2 = quat_to_forward_vector(rotate_quat(data.qpos[3:7], np.pi/8))
            data.mocap_pos = data.qpos[:3] + np.asarray([v2[0], v2[1], 0.0])
            print(f"Step {i} forward: {v}, theta: {np.arctan2(v[1], v[0])}")
            viewer.sync()
            
            # Render video
            if frame_count < data.time * video_fps:
                renderer.update_scene(data, camera="top")
                pixels = renderer.render()
                video.add_image(pixels)
                frame_count += 1
            
            i = i + 1
            print(np.asarray([data.time]).shape, np.array(data.qpos).shape, np.array(data.qvel).shape)
            TRAJ.append(np.concatenate((np.asarray([data.time]),np.array(data.qpos),np.array(data.qvel))))
        TRAJ = np.stack(TRAJ)
        np.save(experiment_folder / "traj.npy", TRAJ)