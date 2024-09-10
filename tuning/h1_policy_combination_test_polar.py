from datetime import datetime
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import mediapy as media
import networkx as nx
import numpy as np
import cv2
import pathlib
import pickle
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
agent = agent_lib.Agent(task_id="H1 Walk", 
                        model=model, 
                        server_binary_path=pathlib.Path(agent_lib.__file__).parent
                        / "mjpc"
                        / "agent_server")
# agent_x.set_cost_weights({'Face goal':0.0, 'Posture up': 0.1})
# agent_y.set_cost_weights({'Face goal':0.0, 'Posture up': 0.1})
agent.set_cost_weights({'Posture arms': 0.06, 'Posture torso': 0.05, 'Face goal': 4.0})

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
    print(f"Finding closest point to {x}")
    x = np.array(x)
    dist = np.linalg.norm(np.stack((fit_x(t) - x[0], fit_y(t) - x[1]), axis=-1), axis=-1)
    min_index = np.argmin(dist)
    tangent_vector = np.array([fit_x_d(min_index), fit_y_d(min_index)])
    tangent_vector /= np.linalg.norm(tangent_vector)
    return np.argmin(dist), dist[min_index], tangent_vector

def crowdsourcing_cost(x,y,forward_vector):
    x = np.array([x,y])
    closest_point, crosstrack_error, tangent_vector = closest_point_on_path(x)
    orientation_error = (np.dot(forward_vector, tangent_vector)-1)**2
    print(f"crosstrack error: {crosstrack_error}, orientation error: {orientation_error}")
    return crosstrack_error + 10*orientation_error

def get_crowdsourcing_costs(data):
    fw = quat_to_forward_vector(data.qpos[3:7])
    fw_left = quat_to_forward_vector(rotate_quat(data.qpos[3:7], np.pi/16))
    fw_right = quat_to_forward_vector(rotate_quat(data.qpos[3:7], -np.pi/16))
    future_pos = data.qpos[:2] + 0.1 * fw
    future_pos_left = data.qpos[:2] + 0.1 * fw_left
    future_pos_right = data.qpos[:2] + 0.1 * fw_right
    costs = {}
    costs["FORWARD"] = crowdsourcing_cost(future_pos[0], future_pos[1], fw)
    costs["LEFT"] = crowdsourcing_cost(future_pos_left[0], future_pos_left[1], fw_left)
    costs["RIGHT"] = crowdsourcing_cost(future_pos_right[0], future_pos_right[1], fw_right)
    print(costs)
    return costs

def get_crowdsourcing_costs_prob(models, data):
    def get_policy_cost(policy):
        curr_vels = data.qvel[:3]
        samples = np.random.multivariate_normal(np.dot(models[policy]['coeffs'], np.append(curr_vels, 1)), models[policy]['cov'], 100)
        future_pos = data.qpos[:2] + 0.002 * samples[:,:2]
        fw_vectors = [quat_to_forward_vector(rotate_quat(data.qpos[3:7], angle)) for angle in samples[:,2]*0.002]
        cost_samples = [crowdsourcing_cost(x[0][0], x[0][1], x[1]) for x in zip(future_pos, fw_vectors)]
        return np.mean(cost_samples)
    costs = {policy: get_policy_cost(policy) for policy in models.keys()}
    return costs

def get_mocap_reference(data, command):
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

plt.show()
input("Press Enter to continue...")

TRAJ = []
command = "FORWARD"

with mujoco.viewer.launch_passive(model, data) as viewer, ThreadPoolExecutor() as executor:
    with media.VideoWriter(video_path, fps=video_fps, shape=video_resolution) as video:
        models = pickle.load(open("/home/antonio/uni/tesi/mujoco_mpc/tuning/models_ols.pkl", "rb"))
        while viewer.is_running():
            command_costs = get_crowdsourcing_costs_prob(models,data)
            command = min(command_costs, key=command_costs.get)
            data.mocap_pos, data.mocap_quat = get_mocap_reference(data, command)
            # set planner state
            agent.set_state(
                time=data.time,
                qpos=data.qpos,
                qvel=data.qvel,
                act=data.act,
                mocap_pos=data.mocap_pos,
                mocap_quat=data.mocap_quat,
                userdata=data.userdata,
            )
            agent.planner_step()
            data.ctrl = agent.get_action(nominal_action=False)
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