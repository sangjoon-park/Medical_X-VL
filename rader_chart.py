import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from math import pi
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

## 데이터 준비
df = pd.DataFrame({
    'Character': ['X-VL (ours)', 'CheXzero', 'ConVIRT', 'GLoRIA', 'BioViL', 'BioViL-T', 'MedCLIP', 'M3AE'],
    # 'Atelectasis': [0.8028, 0.7869, 0.6852, 0.5418, 0.6459, 0.7119, 0.8124, 0.7059],
    # 'Cardiomegaly': [0.8741, 0.8977, 0.6527, 0.6154, 0.7920, 0.8211, 0.9005, 0.7649],
    # 'Consolidation': [0.8801, 0.9041, 0.6630, 0.6305, 0.6829, 0.8518, 0.8683, 0.9131],
    # 'Edema': [0.9028, 0.8883, 0.7256, 0.5753, 0.7570, 0.7960, 0.9144, 0.8647],
    # 'Pleural Effusion': [0.9335, 0.9167, 0.6470, 0.7799, 0.8906, 0.8549, 0.9310, 0.9266]
    'Atelectasis': [0.6656, 0.6414, 0.5596, 0.4787, 0.5531, 0.5915, 0.6572, 0.6082],
    'Cardiomegaly': [0.6921, 0.7480, 0.5380, 0.5217, 0.6216, 0.6467, 0.7506, 0.6241],
    'Consolidation': [0.4134, 0.5145, 0.1719, 0.1809, 0.2259, 0.4190, 0.3947, 0.4963],
    'Edema': [0.6382, 0.6121, 0.3992, 0.3328, 0.4890, 0.5292, 0.6696, 0.5925],
    'Pleural Effusion': [0.7400, 0.7078, 0.4319, 0.5345, 0.6765, 0.6167, 0.7795, 0.7292]
})

## 하나로 합치기
labels = df.columns[1:]
num_labels = len(labels)

angles = [x / float(num_labels) * (2 * pi) for x in range(num_labels)]  ## 각 등분점
angles += angles[:1]  ## 시작점으로 다시 돌아와야하므로 시작점 추가

# my_palette = plt.cm.get_cmap("Set1", len(df.index))
my_palette = ['red', 'blue', 'green', 'gold', 'black', 'blueviolet', 'turquoise', 'gray']

fig = plt.figure(figsize=(8, 8))
fig.set_facecolor('white')
ax = fig.add_subplot(polar=True)
for i, row in df.iterrows():
    color = my_palette[i]
    data = df.iloc[i].drop('Character').tolist()
    data += data[:1]

    ax.set_theta_offset(pi / 2)  ## 시작점
    ax.set_theta_direction(-1)  ## 그려지는 방향 시계방향

    plt.xticks(angles[:-1], labels, fontsize=14)  ## 각도 축 눈금 라벨
    ax.tick_params(axis='x', which='major', pad=15)  ## 각 축과 눈금 사이에 여백을 준다.

    ax.set_rlabel_position(0)  ## 반지름 축 눈금 라벨 각도 설정(degree 단위)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ['0.2', '0.4', '0.6', '0.8'], fontsize=10)  ## 반지름 축 눈금 설정
    # plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    plt.ylim(0.1, 0.8)

    if i == 0:
        l = 3
    else:
        l = 1
    ax.plot(angles, data, color=color, linewidth=l, linestyle='solid', label=row.Character)  ## 레이더 차트 출력
    # ax.fill(angles, data, color=color, alpha=0.1)  ## 도형 안쪽에 색을 채워준다.

plt.legend(loc=(0.9, 0.85))
plt.show()