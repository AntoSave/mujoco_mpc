import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import pathlib
import scipy
import time
from concurrent.futures import ThreadPoolExecutor
from mujoco_mpc import agent as agent_lib
import scipy.stats

from plotting_utils import PlotManager

tracking_model_path = (
        pathlib.Path(__file__).parent
        / "../build/mjpc/tasks/h1/tracking/task.xml"
    )
tracking_model = mujoco.MjModel.from_xml_path(str(tracking_model_path))
tracking_model.opt.timestep = 0.002
walking_model_path = (
        pathlib.Path(__file__).parent
        / "../build/mjpc/tasks/h1/walk/task.xml"
    )
walking_model = mujoco.MjModel.from_xml_path(str(walking_model_path))
walking_model.opt.timestep = 0.002
# data
tracking_data = mujoco.MjData(tracking_model)
walking_data = mujoco.MjData(walking_model)
# agent
walking_agent = agent_lib.Agent(task_id="H1 Walk",
                        model=walking_model, 
                        server_binary_path=pathlib.Path(agent_lib.__file__).parent
                        / "mjpc"
                        / "agent_server")
tracking_agent = agent_lib.Agent(task_id="H1 Tracking", 
                        model=tracking_model, 
                        server_binary_path=pathlib.Path(agent_lib.__file__).parent
                        / "mjpc"
                        / "agent_server")

plot_manager = PlotManager()
w = plot_manager.add_window((13, 2), "Errors")
for i in range(1,26):
    plot_manager.add_plot(w, str(i), (-0.1, 0.1))
w2 = plot_manager.add_window((13, 2), "References")
for i in range(1,26):
    plot_manager.add_plot(w2, str(i), (-0.4, 0.4))
plot_manager.start()
def get_next_reference(data):
    walking_agent.set_state(
        time=data.time,
        qpos=data.qpos,
        qvel=data.qvel,
        act=data.act,
        mocap_pos=np.asarray([[10.0, 0, 0]])
    )
    for _ in range(10):
        walking_agent.planner_step()
    best_trajectory = walking_agent.best_trajectory()
    reference_qpos = best_trajectory["states"][-1, :walking_model.nq]
    reference_time = best_trajectory["times"][-1]
    return reference_time, reference_qpos

horizon = 0.0

steps_per_planning_iteration = 1
i = 0

with mujoco.viewer.launch_passive(tracking_model, tracking_data) as viewer, ThreadPoolExecutor() as executor:
    while viewer.is_running():
        
        if horizon <= tracking_data.time:
            horizon, ref = get_next_reference(tracking_data)
            tracking_data.userdata[0] = horizon
            tracking_data.userdata[1:tracking_model.nq + 1] = ref
            print(f"New reference at {horizon}: {ref}")
        
        # set planner state
        tracking_agent.set_state(
            time=tracking_data.time,
            qpos=tracking_data.qpos,
            qvel=tracking_data.qvel,
            act=tracking_data.act,
            mocap_pos=tracking_data.mocap_pos,
            mocap_quat=tracking_data.mocap_quat,
            userdata=tracking_data.userdata,
        )
        if i % steps_per_planning_iteration == 0:
            t = time.time()
            # f1 = executor.submit(plan_and_get_trajectory, tracking_agent)
            # traj1 = f1.result()
            tracking_agent.planner_step()
            elapsed = time.time() - t
            print("Elapsed planning time: ", elapsed)
        
        tracking_data.ctrl = tracking_agent.get_action(nominal_action=False)
        mujoco.mj_step(tracking_model, tracking_data)
        curr_ref = tracking_agent.get_state().userdata[tracking_model.nq + 1:]
        plot_manager.send_data(w, (tracking_data.time, curr_ref - tracking_data.qpos))
        plot_manager.send_data(w2, (tracking_data.time, curr_ref))
        print(f"Step {i} state: {tracking_data.qpos}")
        viewer.sync()
        i = i + 1