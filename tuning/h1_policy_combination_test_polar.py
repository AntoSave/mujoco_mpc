from datetime import datetime
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import mediapy as media
import networkx as nx
import numpy as np
import cv2
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
agent_forward = agent_lib.Agent(task_id="H1 Walk", 
                        model=model, 
                        server_binary_path=pathlib.Path(agent_lib.__file__).parent
                        / "mjpc"
                        / "agent_server")
agent_right = agent_lib.Agent(task_id="H1 Walk",
                          model=model,
                            server_binary_path=pathlib.Path(agent_lib.__file__).parent
                            / "mjpc"
                            / "agent_server")
agent_left = agent_lib.Agent(task_id="H1 Walk",
                          model=model,
                            server_binary_path=pathlib.Path(agent_lib.__file__).parent
                            / "mjpc"
                            / "agent_server")
# agent_x.set_cost_weights({'Face goal':0.0, 'Posture up': 0.1})
# agent_y.set_cost_weights({'Face goal':0.0, 'Posture up': 0.1})
for agent in [agent_forward, agent_right, agent_left]:
    agent.set_cost_weights({'Posture up': 0.1})

# Experiment info
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_name = f"h1_walk_{current_datetime}"
experiment_folder = pathlib.Path(__file__).parent.parent / "experiments" / experiment_name
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
grad = np.gradient(Z)
ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(X, Y, Z)
ax = plt.figure().add_subplot()
ax.quiver(X, Y, -grad[1], -grad[0], angles='xy')

G = nx.generators.grid_2d_graph(100, 100)

print("Adding weights")
for source,dest,data_dict in G.edges(data=True):
    data_dict['weight'] = cost(np.array([dest[0]/10.0, dest[1]/10.0]))
def dist(a, b):
    (x1, y1) = a
    (x2, y2) = b
    x1/=10.0
    y1/=10.0
    x2/=10.0
    y2/=10.0
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
path = nx.astar_path(G, (0, 0), (99, 99), heuristic=dist, weight="weight")
path_arr = np.array(path) * 0.1

# Plot path as line
plt.plot(path_arr[:,0], path_arr[:,1])

# Plot path as heatmap
# path_xy = np.zeros((100,100), dtype=np.uint8)
# for p in path:
#     path_xy[p[0], p[1]] = 1
# path_xy = cv2.dilate(path_xy, np.ones((3,3), np.uint8))
# plt.imshow(path_xy, origin='lower')
t = np.arange(path_arr.shape[0])
fit_x = np.polynomial.polynomial.Polynomial.fit(t, path_arr[:,0], 10)
fit_y = np.polynomial.polynomial.Polynomial.fit(t, path_arr[:,1], 10)
fit_x_d = fit_x.deriv()
fit_y_d = fit_y.deriv()
plt.plot(fit_x(t), fit_y(t))

def closest_point_on_path(x):
    x = np.array(x)
    dist = np.linalg.norm(np.stack((fit_x(t) - x[0], fit_y(t) - x[1]), axis=-1), axis=-1)
    min_index = np.argmin(dist)
    tangent_vector = np.array([fit_x_d(min_index), fit_y_d(min_index)])
    tangent_vector /= np.linalg.norm(tangent_vector)
    return np.argmin(dist), dist, tangent_vector

def crowdsourcing_cost(x,y,forward_vector):
    x = np.array([x,y])
    closest_point, dist, tangent_vector = closest_point_on_path(x)
    fw_point = np.array([fit_x(closest_point), fit_y(closest_point)]) + 0.5 * forward_vector
    return dist + np.linalg.norm(x-fw_point) + 100 * np.dot(forward_vector, tangent_vector)

plt.show()
input("Press Enter to continue...")

TRAJ = []

with mujoco.viewer.launch_passive(model, data) as viewer, ThreadPoolExecutor() as executor:
    with media.VideoWriter(video_path, fps=video_fps, shape=video_resolution) as video:
        while viewer.is_running():
            
            forward_vector = np.concatenate([quat_to_forward_vector(data.qpos[3:7]), [0]])
            
            # set planner state
            agent_forward.set_state(
                time=data.time,
                qpos=data.qpos,
                qvel=data.qvel,
                act=data.act,
                mocap_pos=data.qpos[:3] + forward_vector * 10.0,
                mocap_quat=data.qpos[3:7],
                userdata=data.userdata,
            )
            agent_left.set_state(
                time=data.time,
                qpos=data.qpos,
                qvel=data.qvel,
                act=data.act,
                mocap_pos=data.qpos[:3],
                mocap_quat=rotate_quat(data.qpos[3:7], np.pi/2),
                userdata=data.userdata,
            )
            agent_right.set_state(
                time=data.time,
                qpos=data.qpos,
                qvel=data.qvel,
                act=data.act,
                mocap_pos=data.qpos[:3],
                mocap_quat=rotate_quat(data.qpos[3:7], -np.pi/2),
                userdata=data.userdata,
            )
            
            if i % steps_per_planning_iteration == 0:
                t = time.time()
                f1 = executor.submit(plan_and_get_trajectory, agent_forward)
                f2 = executor.submit(plan_and_get_trajectory, agent_left)
                f3 = executor.submit(plan_and_get_trajectory, agent_right)
                traj1 = f1.result()
                traj2 = f2.result()
                traj3 = f3.result()
                elapsed = time.time() - t
                print("Elapsed planning time: ", elapsed)
                traj1_states = traj1["states"]
                traj2_states = traj2["states"]
                traj3_states = traj3["states"]
                #euler distance from goal
                traj1_goal_distance = crowdsourcing_cost(traj1_states[-1,0], traj1_states[-1,1], quat_to_forward_vector(traj1_states[-1,3:7]))
                traj2_goal_distance = crowdsourcing_cost(traj2_states[-1,0], traj2_states[-1,1], quat_to_forward_vector(traj2_states[-1,3:7]))
                traj3_goal_distance = crowdsourcing_cost(traj3_states[-1,0], traj3_states[-1,1], quat_to_forward_vector(traj3_states[-1,3:7]))
                current_agent = np.argmin([traj1_goal_distance, traj2_goal_distance, traj3_goal_distance])

            if current_agent == 0:
                agent = agent_forward
            elif current_agent == 1:
                agent = agent_left
            else:
                agent = agent_right
            data.ctrl = agent.get_action(nominal_action=True)
            mujoco.mj_step(model, data)
            print(f"Step {i} state: {data.qpos}")
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
        np.save("traj.npy", TRAJ)