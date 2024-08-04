import dearpygui.dearpygui as dpg
import math
import time
import random
from collections import deque
dpg.create_context()

DEQUE_MAX_LEN = 20000
data_x = deque(maxlen=DEQUE_MAX_LEN)
data_y = deque(maxlen=DEQUE_MAX_LEN)

def generate_data():
    data_x.append(time.time())
    data_y.append(math.sin(data_x[-1]) + random.uniform(-0.1, 0.1))
    return list(data_x), list(data_y)

def update_plot():
    updated_data_x, updated_data_y = generate_data()
    for i in range(10):
        dpg.configure_item(f'line{i}', x=updated_data_x, y=updated_data_y)
        if dpg.get_value("auto_fit_checkbox"):
            dpg.fit_axis_data(f"xaxis{i}")

with dpg.window(tag = "Main Window"):
    with dpg.subplots(rows=5, columns=2, tag="subplots", height=1300, width=1100):
        for i in range(10):
            with dpg.plot(height=200, width=500):
                dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag=f"xaxis{i}", time=True, no_tick_labels=True)
                dpg.add_plot_axis(dpg.mvYAxis, label="Amplitude", tag=f"yaxis{i}")
                dpg.add_line_series([], [], tag=f'line{i}', parent=f"yaxis{i}")
                dpg.set_axis_limits(f"yaxis{i}", -1.5, 1.5)
    dpg.add_checkbox(label="Auto-fit x-axis limits", tag="auto_fit_checkbox", default_value=True)

dpg.create_viewport(width=1100, height=1300, title='Updating plot data')
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Main Window", True)
while dpg.is_dearpygui_running():
    start = time.perf_counter_ns()
    update_plot() # updating the plot directly from the running loop
    dpg.render_dearpygui_frame()
    end = time.perf_counter_ns()
    print(f"Time taken to render frame: {(end - start) / 1e6} ms")
    print(f"FPS: {1 / ((end - start) / 1e9)}")
dpg.destroy_context()