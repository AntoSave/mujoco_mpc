"""A tool for visualizing trajectories in MuJoCo using Dear PyGui."""

import copy
import gymnasium as gym
import dearpygui.dearpygui as dpg
import datetime as dt
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import mujoco.viewer
from mujoco_mpc import agent as agent_lib
import multiprocessing as mp
import multiprocessing.shared_memory as mp_shm
import numpy as np
import os
import pathlib
import time
from PIL import Image


mp_fork = mp.get_context("fork")
mp_spawn = mp.get_context("spawn")

class _View:
    def __init__(self, id, model, trajectory = None, width = 500, height = 500, camera_name = None):
        if not width:
            width = 500
        if not height:
            height = 500
        self._id = id
        self._model_path = model
        self._trajectory = trajectory
        self._current_traj_index = 0
        self._width = width
        self._height = height
        self._camera_name = camera_name
        self._sh_mem = None
        self._raw_data = None
      
    def _initialize_gui(self):
        self._sh_mem = mp_shm.SharedMemory(name=f"texture_{self._id}", create=True, size=self._width*self._height*4*8)
        self._raw_data = np.ndarray(shape=(self._width, self._height, 4), dtype=np.float32, buffer=self._sh_mem.buf)
        np.copyto(self._raw_data, np.ones((self._width, self._height, 4), dtype=np.float32))
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(width=self._width, height=self._height, default_value=self._raw_data, format=dpg.mvFormat_Float_rgba, tag=f"texture_{self._id}")
        dpg.add_image(f"texture_{self._id}", width=self._width, height=self._height)
        
    def _initialize_renderer(self):
        while not self._sh_mem:
            try:
                self._sh_mem = mp_shm.SharedMemory(name=f"texture_{self._id}", create=False, size=self._width*self._height*4*8)
                self._raw_data = np.ndarray(shape=(self._width, self._height, 4), dtype=np.float32, buffer=self._sh_mem.buf)
            except FileNotFoundError:
                print(f"[{mp.current_process().name}] Waiting for shared memory 'texture_{self._id}' to be created")
                time.sleep(0.1)
        self._model = mujoco.MjModel.from_xml_path(str(self._model_path))
        self._data = mujoco.MjData(self._model)
        mujoco.mj_step(self._model, self._data)
        self._renderer = mujoco.Renderer(self._model, width=self._width, height=self._height)
    
    def _update_trajectory(self, trajectory):
        self._trajectory = trajectory
        self._current_traj_index = 0
        
    def _update_image(self):
        if self._trajectory is None:
            return
        self._data.qpos = self._trajectory[self._current_traj_index]
        mujoco.mj_forward(self._model, self._data)
        self._renderer.update_scene(self._data, camera=self._camera_name if self._camera_name else -1)
        pixels = self._renderer.render()
        img = Image.fromarray(pixels) #TODO: Remove PIL dependency to make code faster
        img = img.convert("RGBA")
        dpg_image = np.frombuffer(img.tobytes(), dtype=np.uint8) / 255.0
        dpg_image = np.reshape(dpg_image, (self._width, self._height, 4))
        np.copyto(self._raw_data, dpg_image)
        self._current_traj_index += 1
        if self._current_traj_index == self._trajectory.shape[0]:
            self._current_traj_index = 0
    
    def _debug_print(self):
        print(f"[{mp.current_process().name}] View {self._id}: ", self._raw_data)

class _Window:
    def __init__(self, id, shape: tuple[2]):
        super(_Window, self).__init__()
        self._id = id
        self._next_view_id = 0
        self._grid_shape: tuple[2] = shape
        self._views: list[_View] = []
        print(f"Process name: {mp.current_process().name}")
    
    def add_view(self, model, trajectory = None, width=None, height=None, camera_name = None):
        view = _View(self._next_view_id, model, trajectory, width, height, camera_name)
        self._views.append(view)
        self._next_view_id += 1
        print(f"Added view with id {view._id}")
        return view
    
    def _initialize_gui(self):
        print(f"Process name: {mp.current_process().name}")
        with dpg.window(tag = "Main Window"):
            with dpg.group(horizontal=False):
                for i in range(self._grid_shape[0]):
                    with dpg.group(horizontal=True):
                        for j in range(self._grid_shape[1]):
                            index = i*self._grid_shape[1] + j
                            if index >= len(self._views):
                                break
                            self._views[index]._initialize_gui()
                            
    def _initialize_renderer(self):
        print(f"Process name: {mp.current_process().name}")
        for view in self._views:
            view._initialize_renderer()
    
    def _debug_print(self):
        for view in self._views:
            view._debug_print()
    
    def _update_images(self):
        for view in self._views:
            view._update_image()

class TrajectoryViewer:
    def __init__(self, shape: tuple[2], width = 1000, height = 1000):
        super(TrajectoryViewer, self).__init__()
        self._window = _Window(id=0, shape=shape)
        self._width = width
        self._height = height
    
    def add_view(self, model, trajectory = None, width=None, height=None, camera_name = None):
        return self._window.add_view(model, trajectory, width=width, height=height, camera_name = camera_name)
    
    def start(self):
        self.p_gui = mp_spawn.Process(target=self.run_gui, args=(self._window,self._width, self._height), daemon=True, name="GUI")
        self.p_renderer = mp_spawn.Process(target=self.run_renderer, args=(self._window,), daemon=True, name="Renderer")
        self.p_gui.start()
        self.p_renderer.start()
    
    def join(self):
        self.p_gui.join()
        self.p_renderer.join()
    
    @staticmethod
    def run_gui(window, width=1000, height=1000):
        print("Starting Trajectory Viewer")
        dpg.create_context()
        window._initialize_gui()
        dpg.create_viewport(width=width, height=height, title='Trajectory Viewer')
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Main Window", True)
        while dpg.is_dearpygui_running():
            #start = time.perf_counter_ns()
            #window._debug_print()
            dpg.render_dearpygui_frame()
            #end = time.perf_counter_ns()
            # print(f"Time taken to render frame: {(end - start) / 1e6} ms")
            # print(f"FPS: {1 / ((end - start) / 1e9)}")
        dpg.destroy_context()
        
    @staticmethod
    def run_renderer(window):
        window._initialize_renderer()
        while True:
            window._update_images()
            #window._debug_print()
            time.sleep(0.002)
       
if __name__ == "__main__":
    model_path = (
        pathlib.Path(__file__).parent
        / "../build/mjpc/tasks/h1/walk/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    # data = mujoco.MjData(model)
    # data.qpos = [ 0.02762667,  0.13404553,  1.05540741,  0.99795042, -0.06369025,  0.00204179, 
    # -0.00585946,  0.42735359,  0.00819075,  0.08080127, -0.05984312, -0.11663226, 
    # 0.13752153,  0.18018916, -1.07652994,  1.69882327, -0.62080289, -0.1733220, 
    # -0.07675962,  0.07708435,  0.0069074,  -0.01577377,  0.03256376,  0.0832239,
    # 0.00685365, -0.01660792 ]
    # agent = agent_lib.Agent(task_id="H1 Walk", 
    #                     model=model, 
    #                     server_binary_path=pathlib.Path(agent_lib.__file__).parent
    #                     / "mjpc"
    #                     / "agent_server")
    # agent.set_state(
    #     time=data.time,
    #     qpos=data.qpos,
    #     qvel=data.qvel,
    #     act=data.act,
    #     mocap_pos=np.asarray([10.0, 0.0, 0.25]),
    #     mocap_quat=data.mocap_quat,
    #     userdata=data.userdata,
    # )
    # for i in range(1):
    #     cost = agent.get_total_cost()
    #     print(f"Cost: {cost}")
    #     agent.planner_step()
        
    # traj = agent.best_trajectory()["states"]
    # traj = traj[:, :model.nq]
    # agent.set_state(
    #     time=data.time,
    #     qpos=data.qpos,
    #     qvel=data.qvel,
    #     act=data.act,
    #     mocap_pos=np.asarray([10.0, 0.0, 0.25]),
    #     mocap_quat=data.mocap_quat,
    #     userdata=data.userdata,
    # )
    # for i in range(10):
    #     t = time.time()
    #     agent.planner_step()
    #     print(f"Time taken: {time.time() - t}")
        
    # traj2 = agent.best_trajectory()["states"]
    # traj2 = traj2[:, :model.nq]
    # print(traj.shape)
    traj = np.load("/home/antonio/uni/tesi/mujoco_mpc/experiments/h1_teleop_experiments/absolute_primitives_demo/traj.npy")[:, 1:model.nq+1]
    
    traj1 = traj[100:427, :]
    traj2 = traj[460:799, :]
    traj_viewer = TrajectoryViewer(shape=(1,2))
    view = traj_viewer.add_view(model_path, trajectory=traj1, camera_name="top", width=1000, height=720)
    view2 = traj_viewer.add_view(model_path, trajectory=traj2, camera_name="top", width=1000, height=720)
    traj_viewer.start()
    traj_viewer.join()