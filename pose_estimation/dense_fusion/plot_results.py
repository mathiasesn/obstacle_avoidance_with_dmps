import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors


objs = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
sigmas = [0.0, 0.001, 0.005, 0.01, 0.02, 0.03]
obj_names = [
    'Ape', 'Bench vise', 'Can', 'Camera', 'Cat',
    'Drill', 'Duck', 'Egg box', 'Glue', 'Hole pouncher',
    'Iron', 'Lamp', 'Phone', 'All'
]

file_paths = [
    'pose_estimation/dense_fusion/results/linemod/eval_result_0.0_logs.txt',
    'pose_estimation/dense_fusion/results/linemod/eval_result_0.001_logs.txt',
    'pose_estimation/dense_fusion/results/linemod/eval_result_0.005_logs.txt',
    'pose_estimation/dense_fusion/results/linemod/eval_result_0.01_logs.txt',
    'pose_estimation/dense_fusion/results/linemod/eval_result_0.02_logs.txt',
    'pose_estimation/dense_fusion/results/linemod/eval_result_0.03_logs.txt'
]

data_objs = np.empty((len(sigmas), len(objs), 3))
data_all = np.empty((len(sigmas), 3))

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = [mpl_colors.to_rgb(c) for c in prop_cycle.by_key()['color']]

for i, file_path in enumerate(file_paths):
    print(f'Current file path: {file_path}')
    f = open(file_path, 'r')

    while True:
        line = f.readline()

        if line[0] == 'N':
            continue

        if not line:
            break

        line = line.split()
        if line[0] == 'Object':
            acc = float(line[4])
            dist = float(line[6])
            std = float(line[8])
            index = objs.index(int(line[1]))
            data_objs[i, index, :] = (acc, dist, std)
        
        if line[0] == 'ALL':
            acc = float(line[3])
            line = f.readline().split()
            dist = float(line[1])
            std = float(line[3])
            data_all[i, :] = (acc, dist, std)
            break

plt.plot(sigmas, data_objs[:, :, 0])
plt.ylabel('Accuracy')
plt.xlabel('Standard deviation of added noise')
plt.legend(obj_names)
plt.grid('on')
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 1)
axs[0].plot(sigmas, data_all[:, 0])
axs[0].set_ylabel('Accuracy')
axs[0].grid(True)
plt.setp(axs[0].get_xticklabels(), visible=False)

axs[1].errorbar(sigmas, data_all[:, 1], yerr=data_all[:, 2], capsize=8)
axs[1].set_ylabel('Distance')
axs[1].set_xlabel('Standard deviation of added noise')
axs[1].grid(True)

fig.tight_layout()
plt.show()