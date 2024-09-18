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

def cost(x, goal=np.array([10.0, 10.0]), obstacles=np.array([[3.0, 3.0]])):
    """Cost function for the planner. x is a two dimensional state vector."""
    c = np.linalg.norm(x - goal)
    for o in obstacles:
        c += 30*scipy.stats.multivariate_normal.pdf(x, mean=o, cov=0.4*np.eye(2))
    return c

def export_cost_plots(cost_function, directory_path = pathlib.Path(__file__).parent):
    X, Y = np.meshgrid(np.linspace(-1, 11, 100), np.linspace(-1, 11, 100))
    z = np.array([cost_function(np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = z.reshape(X.shape)
    grad = np.gradient(Z)
    fig_3d = plt.figure()
    ax = fig_3d.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z)
    fig_3d.savefig(directory_path / "cost_3d.png")
    fig_grad = plt.figure()
    ax = fig_grad.add_subplot()
    ax.quiver(X, Y, -grad[1], -grad[0], angles='xy')
    fig_grad.savefig(directory_path / "cost_3d_grad.png")
    return fig_3d, fig_grad

def get_path(starting_point, goal_point, cost_func, extreme_points=np.array([[0.0,0.0],[10.0,10.0]]), n_bins = 100):
    space_lengths = extreme_points[1] - extreme_points[0]
    step_sizes = space_lengths / n_bins
    starting_node = tuple(np.floor(starting_point / step_sizes).astype(int))
    goal_node = tuple(np.floor(goal_point / step_sizes).astype(int))
    G = nx.generators.grid_2d_graph(n_bins+1, n_bins+1)
    for source,dest,data_dict in G.edges(data=True):
        data_dict['weight'] = cost_func(np.array([dest[0]*step_sizes[0], dest[1]*step_sizes[1]]))
    def dist(a, b):
        (x1, y1) = a
        (x2, y2) = b
        x1 *= step_sizes[0]
        y1 *= step_sizes[1]
        x2 *= step_sizes[0]
        y2 *= step_sizes[1]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    path = nx.astar_path(G, starting_node, goal_node, heuristic=dist, weight="weight")
    path_arr = np.array(path) * step_sizes
    return path_arr

# Plot path as heatmap
# path_xy = np.zeros((100,100), dtype=np.uint8)
# for p in path:
#     path_xy[p[0], p[1]] = 1
# path_xy = cv2.dilate(path_xy, np.ones((3,3), np.uint8))
# plt.imshow(path_xy, origin='lower')
def fit_polynomial(path_arr, degree=10):
    t = np.arange(path_arr.shape[0])
    fit_x = np.polynomial.polynomial.Polynomial.fit(t, path_arr[:,0], degree)
    fit_y = np.polynomial.polynomial.Polynomial.fit(t, path_arr[:,1], degree)
    fit_x_d = fit_x.deriv()
    fit_y_d = fit_y.deriv()
    return t, fit_x, fit_y, fit_x_d, fit_y_d

def ARMA_filter(path, p = 3, q = 3):
    smoothed_path = np.zeros_like(path)
    for i, point in enumerate(path):
        smoothed_path[i] = np.mean(path[max(0, i-p):min(i+q, path.shape[0])], axis=0)
    return smoothed_path

def path_tangent_vectors(path):
    path_diff = np.diff(path, axis=0)
    path_diff = path_diff / np.linalg.norm(path_diff, axis=1)[:,None]
    return path_diff

def closest_point_on_path(x, path, path_d):
    x = np.array(x)
    dist = np.linalg.norm(x-path, axis=-1)
    min_index = np.argmin(dist)
    tangent_vector = path_d[min_index]
    return min_index, dist[min_index], tangent_vector

def crowdsourcing_cost(x, y, forward_vector, path, path_d):
    x = np.array([x,y])
    closest_point, crosstrack_error, tangent_vector = closest_point_on_path(x, path, path_d)
    orientation_error = (np.dot(forward_vector, tangent_vector)-1)**2
    print(f"Cross-track error: {crosstrack_error}, Orientation error: {orientation_error}")
    return crosstrack_error + 10*orientation_error

def get_crowdsourcing_costs(data, path, path_d):
    fw = quat_to_forward_vector(data.qpos[3:7])
    fw_left = quat_to_forward_vector(rotate_quat(data.qpos[3:7], np.pi/16))
    fw_right = quat_to_forward_vector(rotate_quat(data.qpos[3:7], -np.pi/16))
    future_pos = data.qpos[:2] + 0.1 * fw
    future_pos_left = data.qpos[:2] + 0.1 * fw_left
    future_pos_right = data.qpos[:2] + 0.1 * fw_right
    costs = {}
    costs["FORWARD"] = crowdsourcing_cost(future_pos[0], future_pos[1], fw, path, path_d)
    costs["LEFT"] = crowdsourcing_cost(future_pos_left[0], future_pos_left[1], fw_left, path, path_d)
    costs["RIGHT"] = crowdsourcing_cost(future_pos_right[0], future_pos_right[1], fw_right, path, path_d)
    print(costs)
    return costs

def get_crowdsourcing_costs_prob(models, data, path, path_d):
    def get_policy_cost(policy):
        curr_vels = data.qvel[[0,1,5]]
        samples = np.random.multivariate_normal(np.dot(models[policy]['coeffs'], np.append(curr_vels, 1)), models[policy]['cov'], 10000)
        future_pos = data.qpos[:2] + 0.002 * samples[:,:2]
        fw_vectors = [quat_to_forward_vector(rotate_quat(data.qpos[3:7], angle)) for angle in np.atan2(samples[:,1],samples[:,0])*0.002]
        cost_samples = [crowdsourcing_cost(x[0][0], x[0][1], x[1], path, path_d) for x in zip(future_pos, fw_vectors)]
        return np.mean(cost_samples)
    costs = {policy: get_policy_cost(policy) for policy in models.keys()}
    return costs

def get_fno_from_delta(qpos, qpos_old, delta_time):
    vx = (qpos[0] - qpos_old[0])/delta_time
    vy = (qpos[1] - qpos_old[1])/delta_time
    fw = quat_to_forward_vector(qpos[3:7])
    theta = np.arctan2(fw[1], fw[0])
    v_fw = np.dot([vx, vy], np.array([np.cos(theta), np.sin(theta)]))
    v_n = np.dot([vx, vy], np.array([np.cos(theta+np.pi/2), np.sin(theta+np.pi/2)]))
    fw_old = quat_to_forward_vector(qpos_old[3:7])
    theta_old = np.arctan2(fw_old[1], fw_old[0])
    omega = (theta - theta_old)/delta_time
    return np.array([v_fw, v_n, omega])
    

def get_crowdsourcing_costs_prob_v2(models, data, curr_fno, old_fno, path, path_d):
    """fno is [v_fw, f_n, omega]"""
    fw = quat_to_forward_vector(data.qpos[3:7])
    theta = np.arctan2(fw[1], fw[0])
    lookahead_steps = 50
    timestep = models['timestep']*lookahead_steps
    def get_policy_cost(policy):
        print(f"Policy: {policy}")
        policy_coeffs = models['models'][policy]['coeffs']
        policy_cov = models['models'][policy]['cov']
        regressors = np.array([curr_fno[0], old_fno[0], curr_fno[1], old_fno[1], curr_fno[2], old_fno[2], 1])
        means = np.matmul(policy_coeffs, regressors)
        samples = np.random.multivariate_normal(means, policy_cov/10000, 100)
        future_pos_x = data.qpos[0] + samples[:,0]*np.cos(theta)*timestep*10 + samples[:,1]*np.sin(theta)*timestep*10
        future_pos_y = data.qpos[1] + samples[:,0]*np.sin(theta)*timestep*10 - samples[:,1]*np.cos(theta)*timestep*10
        fw_vectors = [quat_to_forward_vector(rotate_quat(data.qpos[3:7], angle)) for angle in samples[:,2]*timestep]
        cost_samples = [crowdsourcing_cost(x[0], x[1], x[2], path, path_d) for x in zip(future_pos_x, future_pos_y, fw_vectors)]
        return np.mean(cost_samples)
    costs = {policy: get_policy_cost(policy) for policy in models['models'].keys()}
    return costs

# def sample_varmax(model_params, Y, X):
#     L1 = model_params['L1']
#     L2 = model_params['L2']
#     L3 = model_params['L3']
#     L1_e = model_params['L1_e']
#     beta = model_params['beta']
#     intercept = model_params['intercept']
#     sigma2 = model_params['sigma2']
    
#     epsilon_old = np.random.multivariate_normal(np.zeros(3), np.diag(sigma2), 100)
#     epsilon = np.random.multivariate_normal(np.zeros(3), np.diag(sigma2), 100)
    
#     # Y dim is 3x3
#     mean = intercept
#     mean += np.dot(L1, Y[0])
#     mean += np.dot(L2, Y[1])
#     mean += np.dot(L3, Y[2])
#     mean += np.dot(beta, X)
#     return mean + np.squeeze(np.matmul(L1_e,np.expand_dims(epsilon_old, axis=-1))) + epsilon

# def get_crowdsourcing_costs_varmax(models, data, last_increments, path, path_d):
#     def get_policy_cost(policy):
#         curr_vels = data.qvel[[0,1,5]]
#         samples_increments = sample_varmax(models[policy], last_increments, curr_vels)
#         samples = curr_vels + samples_increments # Model is in increments
#         future_pos = data.qpos[:2] + 0.002 * samples[:,:2]
#         fw_vectors = [quat_to_forward_vector(rotate_quat(data.qpos[3:7], angle)) for angle in samples[:,2]*0.002]
#         cost_samples = [crowdsourcing_cost(x[0][0], x[0][1], x[1], path, path_d) for x in zip(future_pos, fw_vectors)]
#         return np.mean(cost_samples)
#     costs = {policy: get_policy_cost(policy) for policy in models.keys()}
#     return costs

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

def add_obstacles(model_spec, obstacles):
    for i, o in enumerate(obstacles):
        body = model_spec.worldbody.add_body()
        body.name = f"obstacle_{i}"
        body.pos = o.tolist() + [0.5]
        geom = body.add_geom()
        geom.name = f"obstacle_geom_{i}"
        geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        geom.size = [0.8, 0.8, 0.5]
        geom.rgba = [1, 0, 0, 1]

if __name__ == "__main__":
    goal = np.array([10.0, 10.0])
    obstacles = np.array([[3.0, 3.0], [5.0, 5.0]])
    cost_function = lambda x: cost(x, goal=goal, obstacles=obstacles)
    
    model_path = (
        pathlib.Path(__file__).parent
        / "../build/mjpc/tasks/h1/walk/task.xml"
    )
    #model = mujoco.MjModel.from_xml_path(str(model_path))
    model_spec = mujoco.MjSpec()
    model_spec.from_file(str(model_path))
    add_obstacles(model_spec, obstacles)
    model = model_spec.compile()
    model.opt.timestep = 0.002
    # data
    data = mujoco.MjData(model)
    # agents
    agent = agent_lib.Agent(task_id="H1 Walk", 
                            model=model, 
                            server_binary_path=pathlib.Path(agent_lib.__file__).parent
                            / "mjpc"
                            / "agent_server")
    agent.set_cost_weights({'Posture arms': 0.06, 'Posture torso': 0.05, 'Face goal': 4.0})

    # Experiment info
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"h1_walk_{current_datetime}"
    experiment_folder = pathlib.Path(__file__).parent.parent / "experiments" / "relative_policy_combination" / experiment_name
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
    
    path_arr = get_path(np.array([0.0,0.0]), np.array([10.0,10.0]), cost_function)
    #t, fit_x, fit_y, fit_x_d, fit_y_d = fit_polynomial(path_arr)
    smoothed_path = ARMA_filter(path_arr, p=3, q=3)
    path_d = path_tangent_vectors(smoothed_path)
    
    path_data = {"path": path_arr, "smoothed_path": smoothed_path, "path_d": path_d, "goal": goal, "obstacles": obstacles}
    pickle.dump(path_data, open(experiment_folder / "path_data.pkl", "wb"))
    
    _, fig_grad = export_cost_plots(cost_function, experiment_folder)
    fig_grad.axes[0].plot(path_arr[:,0], path_arr[:,1])
    fig_grad.axes[0].plot(smoothed_path[:,0], smoothed_path[:,1])
    fig_grad.savefig(experiment_folder / "path.png")
    
    input("Press Enter to continue...")

    TRAJ = []
    command = "FORWARD"
    last_fno = np.zeros((2,3))
    qpos_old = data.qpos[:7][:]
    with mujoco.viewer.launch_passive(model, data) as viewer, ThreadPoolExecutor() as executor:
        with media.VideoWriter(video_path, fps=video_fps, shape=video_resolution) as video:
            models = pickle.load(open("/home/antonio/uni/tesi/mujoco_mpc/experiments/h1_teleop_experiments/2024-09-09_16-10-24/models_ols_fno.pkl", "rb"))
            model_timestep = models['timestep']
            downsample_factor = int(model_timestep / 0.002)
            print(f"Model timestep: {model_timestep}")
            while viewer.is_running():
                #command_costs = get_crowdsourcing_costs_prob(models, data, smoothed_path, path_d)
                #command_costs = get_crowdsourcing_costs(data, smoothed_path, path_d)
                # print("last data", last_data.shape)
                # last_data = np.append(last_data, np.array([[data.qvel[0], data.qvel[1], data.qvel[5]]]), axis=0)
                # last_data = np.delete(last_data, 0, axis=0)
                # print("last data", last_data)
                # last_increments = np.diff(last_data, axis=0)
                # print("last increments", last_increments)
                # command_costs = get_crowdsourcing_costs_varmax(models, data, last_increments[-3:], smoothed_path, path_d)
                if i % downsample_factor == 0 or True:
                    command_costs = get_crowdsourcing_costs_prob_v2(models, data, last_fno[1], last_fno[0], smoothed_path, path_d)
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
                viewer.sync()
                
                # Update fno
                if i % downsample_factor == 0:
                    last_fno[1] = last_fno[0]
                    last_fno[0] = get_fno_from_delta(data.qpos, qpos_old, model_timestep)
                    qpos_old = data.qpos[:7][:]
                
                # Render video
                if frame_count < data.time * video_fps:
                    renderer.update_scene(data, camera="top")
                    pixels = renderer.render()
                    video.add_image(pixels)
                    frame_count += 1
                
                i = i + 1
                TRAJ.append(np.concatenate((np.asarray([data.time]),np.array(data.qpos),np.array(data.qvel))))
            TRAJ = np.stack(TRAJ)
            np.save(experiment_folder / "traj.npy", TRAJ)