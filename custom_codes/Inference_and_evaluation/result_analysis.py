#%matplotlib widget
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

# 예제 데이터: 243 프레임에 대한 임의의 3D 포즈 데이터
# 여기서는 간단한 예시로 각 포즈가 3개의 점으로 구성되어 있으며, 각 점은 (x, y, z) 좌표를 가집니다.
# 실제 데이터에 맞게 조정해주세요.
data = np.random.rand(243, 3, 3)  # (프레임 수, 점의 수, 좌표)

# 초기 프레임 설정
initial_frame = 0

# 그래프 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.25, bottom=0.25)

# 슬라이더를 위한 축
axcolor = 'lightgoldenrodyellow'
axframe = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

# 슬라이더 초기화
frame_slider = Slider(ax=axframe, label='Frame', valmin=0, valmax=242, valinit=initial_frame, valfmt='%0.0f')

# 포즈를 그리는 함수
def update(val):
    frame = int(frame_slider.val)
    ax.clear()
    ax.scatter(data[frame, :, 0], data[frame, :, 1], data[frame, :, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.draw()

# 슬라이더 값이 변경될 때마다 update 함수 호출
frame_slider.on_changed(update)

# 초기 포즈 표시
update(initial_frame)

plt.show()
