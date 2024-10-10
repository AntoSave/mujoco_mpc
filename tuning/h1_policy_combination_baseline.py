"""Test script for the obstacle avoidance task with polar motion services"""

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

def cost(x, goal=np.array([10.0, 10.0]), obstacles={'cylinders': np.array([[3.0, 3.0]])}, obstacles_data={'transition_end': False}):
    """Cost function for the planner. x is a two dimensional state vector."""
    cylinder_obs = obstacles['cylinders']
    swall_obs = obstacles['sliding_walls']
    c = np.linalg.norm(x - goal)
    for o in cylinder_obs:
        c += 30*scipy.stats.multivariate_normal.pdf(x, mean=o, cov=0.4*np.eye(2))
    for o in swall_obs:
        start_point, end_point = o[0], o[1]
        length = np.linalg.norm(end_point-start_point)
        n_obs = max(1,int(length / 1.0))
        obs_perc = np.linspace(0, 1, n_obs)
        if obstacles_data['transition_end']:
            start_point = start_point + o[2]
            end_point = end_point + o[2]
        for perc in obs_perc:
            obs_point = start_point + perc * (end_point - start_point)
            c += 100*scipy.stats.multivariate_normal.pdf(x, mean=obs_point, cov=0.2*np.eye(2))
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
    """Gets a path from starting_point to goal_point using A*"""
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
    """Fits a polynomial of degree 'degree' to the path"""
    t = np.arange(path_arr.shape[0])
    fit_x = np.polynomial.polynomial.Polynomial.fit(t, path_arr[:,0], degree)
    fit_y = np.polynomial.polynomial.Polynomial.fit(t, path_arr[:,1], degree)
    fit_x_d = fit_x.deriv()
    fit_y_d = fit_y.deriv()
    return t, fit_x, fit_y, fit_x_d, fit_y_d

def MA_filter(path, p = 3, q = 3):
    """Applies a MA filter to the path to smooth it"""
    extended_path = np.concatenate([path[0]*np.ones((p,2)), path, path[-1]*np.ones((q,2))], axis=0)
    smoothed_path = np.zeros_like(extended_path)
    for i in range(extended_path.shape[0]):
        smoothed_path[i] = np.mean(extended_path[max(0, i-p):min(i+q, extended_path.shape[0])], axis=0)
    path_diff = np.diff(smoothed_path, axis=0)
    distances = np.linalg.norm(path_diff, axis=1)
    critical_points = np.argwhere(distances < 0.03) + 1
    smoothed_path = np.delete(smoothed_path, critical_points, axis=0)
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

def get_command(pos, fw_vect, path, path_d):
    p_index, p_dist, p_tangent = closest_point_on_path(pos, path, path_d)
    if p_dist > 0.4:
        right_vector = np.array([p_tangent[1], -p_tangent[0]])
        pos_vector = pos - path[p_index]
        is_right = np.dot(right_vector, pos_vector) > 0
        if is_right:
            return "LEFT"
        else:
            return "RIGHT"
    else:
        angle = np.arctan2(fw_vect[1], fw_vect[0]) - np.arctan2(p_tangent[1], p_tangent[0])
        if angle > np.pi:
            angle = angle - 2*np.pi
        if angle < -np.pi:
            angle = angle + 2*np.pi
        if angle > 0.1:
            return "RIGHT"
        elif angle < -0.1:
            return "LEFT"
        else:
            return "FORWARD"
        
def get_command_crosstrack_only(pos, fw_vect, path, path_d):
    def rotate_vector(vector, angle):
        a = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return np.dot(a, vector)
    l_vector = rotate_vector(fw_vect, np.pi/4)
    r_vector = rotate_vector(fw_vect, -np.pi/4)
    t = 1.0
    fw_pos = pos + fw_vect * t
    l_pos = pos + l_vector * t
    r_pos = pos + r_vector * t
    _, fw_dist, _ = closest_point_on_path(fw_pos, path, path_d)
    _, l_dist, _ = closest_point_on_path(l_pos, path, path_d)
    _, r_dist, _ = closest_point_on_path(r_pos, path, path_d)
    dists = np.array([fw_dist, l_dist, r_dist])
    min_index = np.argmin(dists)
    if min_index == 0:
        return "FORWARD"
    elif min_index == 1:
        return "LEFT"
    else:
        return "RIGHT"
    

def get_command_heading_only(pos, fw_vect, path, path_d):
    p_index, p_dist, p_tangent = closest_point_on_path(pos, path, path_d)
    angle = np.arctan2(fw_vect[1], fw_vect[0]) - np.arctan2(p_tangent[1], p_tangent[0])
    if angle > np.pi:
        angle = angle - 2*np.pi
    if angle < -np.pi:
        angle = angle + 2*np.pi
    if angle > 0.1:
        return "RIGHT"
    elif angle < -0.1:
        return "LEFT"
    else:
        return "FORWARD"        

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
    start = np.array([0.0, 0.0])
    goal = np.array([10.0, 10.0])
    
    obstacles = {
        "cylinders": np.array([[3.0, 3.0], [5.0, 5.0]]),
        "sliding_walls": np.array([])
            # np.array([[[3.33,0.0], [3.33,3.33], [0.0,3.33]],
            #                        [[6.66,0.0], [6.66,3.33], [0.0,3.33]],
            #                        [[0.0, 6.66], [3.33, 6.66], [3.33, 0.0]]])#np.array([[[5.0, 5.0], [10.0, 5.0], [-5.0, 0.0]]]), #[[[start_x, start_y], [end_x, end_y], [translation_x, translation_y]]] #
    }
    obstacles_data = {"sliding_walls": np.array([]), "trigger": False, "transition_end": False}
    cost_function = lambda x: cost(x, goal=goal, obstacles=obstacles, obstacles_data=obstacles_data)
    
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
    data.qpos[:2] = start
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
    experiment_folder = pathlib.Path(__file__).parent.parent / "experiments" / "relative_policy_combination_baseline" / experiment_name
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
    
    path_arr = get_path(data.qpos[:2], goal, cost_function)
    #t, fit_x, fit_y, fit_x_d, fit_y_d = fit_polynomial(path_arr)
    smoothed_path = MA_filter(path_arr, p=15, q=15)
    path_d = path_tangent_vectors(smoothed_path)
    print(smoothed_path)
    path_updated = False
    
    path_data = {"path": path_arr, "smoothed_path": smoothed_path, "path_d": path_d, "goal": goal, "obstacles": obstacles}
    pickle.dump(path_data, open(experiment_folder / "path_data.pkl", "wb"))
    
    _, fig_grad = export_cost_plots(cost_function, experiment_folder)
    fig_grad.axes[0].plot(path_arr[:,0], path_arr[:,1])
    fig_grad.axes[0].plot(smoothed_path[:,0], smoothed_path[:,1])
    fig_grad.savefig(experiment_folder / "path.png")
    
    input("Press Enter to continue...")

    TRAJ = []
    command = "FORWARD"
    with mujoco.viewer.launch_passive(model, data) as viewer, ThreadPoolExecutor() as executor:
        with media.VideoWriter(video_path, fps=video_fps, shape=video_resolution) as video:
            while viewer.is_running():
                fw_vect = quat_to_forward_vector(data.qpos[3:7])
                command = get_command(data.qpos[:2], fw_vect, smoothed_path, path_d)
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
                
                # Update dynamic obstacles
                if data.qpos[1] > 2.5:
                    obstacles_data['trigger'] = True
                update_dynamic_obs(obstacles, obstacles_data, model)
                if obstacles_data['transition_end'] and not path_updated:
                    # Update path
                    print("Updating path")
                    path_arr = get_path(data.qpos[:2], goal, cost_function)
                    smoothed_path = MA_filter(path_arr, p=15, q=15)
                    path_d = path_tangent_vectors(smoothed_path)
                    path_updated = True
                
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