import copy
import datetime as dt
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import mujoco.viewer
import numpy as np
import pathlib
import time
import cv2

from mujoco_mpc import agent as agent_lib

def main():
    i = 0
    steps_per_planning_iteration = 1
    video_duration = 1
    video_fps = 60
    video_resolution = (720, 1280)
    frame_count = 0
    goals_reached = 0
    
    # model
    model_path = (
        pathlib.Path(__file__).parent
        / "../build/mjpc/tasks/h1/walk/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = 0.002
    # data
    data = mujoco.MjData(model)
    # renderer
    renderer = mujoco.Renderer(model, height=video_resolution[0], width=video_resolution[1])
    # agent
    agent = agent_lib.Agent(task_id="H1 Walk", 
                            model=model, 
                            server_binary_path=pathlib.Path(agent_lib.__file__).parent
                            / "mjpc"
                            / "agent_server")

    # weights
    #agent.set_cost_weights({"Heigth": 0, "Velocity": 0.15})
    print("Cost weights:", agent.get_cost_weights())

    # parameters
    agent.set_task_parameter("Torso", 1.3)
    agent.set_task_parameter("Speed", 0.7)
    print("Parameters:", agent.get_task_parameters())

    # rollout
    mujoco.mj_resetData(model, data)
    renderer.update_scene(data, camera="top")
    
    # mocap
    mocap_path = [np.asarray([2.0, 2.0, 0.25]),
                  np.asarray([2.0, -2.0, 0.25]), 
                  np.asarray([-2.0, -2.0, 0.25]), 
                  np.asarray([-2.0, 2.0, 0.25])]
    current_mocap = 0
    data.mocap_pos[0] = mocap_path[current_mocap]
    with media.VideoWriter("h1_walk_3.mp4", fps=video_fps, shape=video_resolution) as video:
        while goals_reached < 4:
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
                agent.planner_step()
            
            data.ctrl = agent.get_action(nominal_action=True)
            
            mujoco.mj_step1(model, data)
            mujoco.mj_step2(model, data)
            
            # render
            if frame_count < data.time * video_fps:
                renderer.update_scene(data, camera="top")
                pixels = renderer.render()
                video.add_image(pixels)
                frame_count += 1
            
            # update target if goal reached
            if np.linalg.norm(data.sensor('torso_position').data[:2] - data.mocap_pos[0][:2]) < 0.1:
                current_mocap = (current_mocap + 1) % len(mocap_path)
                data.mocap_pos[0] = mocap_path[current_mocap]
                goals_reached+=1
            
            i+=1
            print(f"Frame {frame_count}, Time {data.time}, Goals reached {goals_reached}")
if __name__ == "__main__":
    main()