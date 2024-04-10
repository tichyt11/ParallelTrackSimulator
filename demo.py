import numpy as np
from SimpleSim import TrackSimulator
import warp as wp

wp.init()


num_robots = 2  # number of simulated robots (in parallel)
T = 2000  # number of simulation timesteps to perform
device = "cpu"

shp = (20, 20)
np_heightmaps = [np.zeros(shp, dtype=np.float32) for _ in range(num_robots)]
np_kfs = [0.5 * np.ones(shp, dtype=np.float32) for _ in range(num_robots)]
res = [0.3 for _ in range(num_robots)]

init_poses = np.zeros((num_robots, 7))
init_poses[:, 2] = 1.5  # z coordinate
init_poses[:, 6] = 1.0  # quaternion w

track_vels = np.ones((num_robots, T, 2))
flipper_angles = np.zeros((num_robots, T, 4))
flipper_angles[0, :, 0] = 0.5

simulator = TrackSimulator(np_heightmaps, np_kfs, res, T, use_renderer=True, device=device)

simulator.set_control(track_vels, flipper_angles)
simulator.set_init_poses(init_poses)

body_q = simulator.simulate(render=True, use_graph=False)

print('body_q: ', body_q.numpy())

