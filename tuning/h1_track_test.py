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
w1 = plot_manager.add_window((14, 2), "Position Errors")
for i in range(0,26):
    plot_manager.add_plot(w1, str(i), (-0.1, 0.1))
w2 = plot_manager.add_window((14, 2), "Position References")
for i in range(0,26):
    plot_manager.add_plot(w2, str(i), (-0.4, 0.4))
w3 = plot_manager.add_window((14, 2), "Velocity Errors")
for i in range(0,25):
    plot_manager.add_plot(w3, str(i), None)
w4 = plot_manager.add_window((14, 2), "Velocity References")
for i in range(0,25):
    plot_manager.add_plot(w4, str(i), None)
plot_manager.start()
def get_next_reference(data, planning_steps=4, lookahead_index=3, repetitions = 4):
    reference_qpos = data.qpos
    reference_qvel = data.qvel
    reference_time = data.time
    for _ in range(repetitions):
        walking_agent.set_state(
            time=reference_time,
            qpos=reference_qpos,
            qvel=reference_qvel,
            act=data.act,
            mocap_pos=np.asarray([[reference_qpos[0]+10.0, 0, 0]])
        )
        for _ in range(planning_steps):
            walking_agent.planner_step()
        best_trajectory = walking_agent.best_trajectory()
        best_states = best_trajectory["states"]
        reference_qpos = best_states[lookahead_index, :walking_model.nq]
        reference_qvel = best_states[lookahead_index, walking_model.nq:walking_model.nq + walking_model.nv]
        reference_time = best_trajectory["times"][lookahead_index]
    return reference_time, reference_qpos, reference_qvel

horizon = 0.0

steps_per_planning_iteration = 1
i = 0

print(f"Current pos at {tracking_data.time}: {tracking_data.qpos[:3]}")
horizon, ref, vel = get_next_reference(tracking_data)
tracking_data.userdata[0] = horizon
tracking_data.userdata[1:tracking_model.nq + 1] = ref
tracking_data.userdata[1 + tracking_model.nq : 1 + tracking_model.nq + tracking_model.nv] = vel
print(f"New reference at {horizon}: {ref[:3]}")

with mujoco.viewer.launch_passive(tracking_model, tracking_data) as viewer, ThreadPoolExecutor() as executor:
    while viewer.is_running():
        if horizon < tracking_data.time:
            #print(f"qpos error at {tracking_data.time}: {ref-tracking_data.qpos}")
            #print(f"qvel error at {tracking_data.time}: {vel-tracking_data.qvel}")
            #print(f"Current pos at {tracking_data.time}: {tracking_data.qpos[:3]}")
            horizon, ref, vel = get_next_reference(tracking_data)
            tracking_data.userdata[0] = horizon
            tracking_data.userdata[1:tracking_model.nq + 1] = ref
            tracking_data.userdata[1 + tracking_model.nq : 1 + tracking_model.nq + tracking_model.nv] = vel
            #print(f"New reference at {horizon}: {ref[:3]}")
        
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
        tracking_agent.set_cost_weights({'Robot Velocity': 0.0, 'Joint Velocities': 0.0, 'Joint Positions': 5.0})
        if i % steps_per_planning_iteration == 0:
            t = time.time()
            # f1 = executor.submit(plan_and_get_trajectory, tracking_agent)
            # traj1 = f1.result()
            tracking_agent.planner_step()
            elapsed = time.time() - t
            #print("Elapsed planning time: ", elapsed)
        tracking_data.ctrl = tracking_agent.get_action(nominal_action=False)
        mujoco.mj_step(tracking_model, tracking_data)
        
        curr_ref = tracking_agent.get_state().userdata[1 + tracking_model.nq + tracking_model.nv:]
        curr_ref_qpos = curr_ref[:tracking_model.nq]
        curr_ref_qvel = curr_ref[tracking_model.nq:]
        # Set the integrators to the reference
        #tracking_data.qpos[:3] = curr_ref_qpos[:3]
        #tracking_data.qvel[:3] = curr_ref_qvel[:3]
        # tracking_data.qpos = curr_ref_qpos
        # tracking_data.qvel = curr_ref_qvel
        # # tracking_data.qpos = ref
        # # tracking_data.qvel = vel
        # mujoco.mj_forward(tracking_model, tracking_data)
        viewer.sync()
        plot_manager.send_data(w1, (tracking_data.time, curr_ref_qpos - tracking_data.qpos))
        plot_manager.send_data(w2, (tracking_data.time, curr_ref_qpos))
        plot_manager.send_data(w3, (tracking_data.time, curr_ref_qvel - tracking_data.qvel))
        plot_manager.send_data(w4, (tracking_data.time, curr_ref_qvel))
        #print(f"Step {i} state: {tracking_data.qpos}")
        
        i = i + 1
        #tracking_data.time += tracking_model.opt.timestep
        #tracking_data.time = horizon 
        # if (tracking_data.time <= horizon):
        #     #time.sleep(horizon - tracking_data.time)
        #     tracking_data.time = horizon