import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import matplotlib.animation as animation
from numba import njit
from matplotlib.animation import FuncAnimation
from matplotlib.animation import HTMLWriter
# 定义解析解函数
def analytical_solution(y, t, D=1.0, sum_terms=50):
    if t == 0:
        sol = np.zeros(len(y))
        sol[-1] = 1  # 上边界条件
        return sol
    
    sol = np.zeros(len(y))
    for i in range(sum_terms):
        sol += erfc((1 - y + 2 * i) / (2 * np.sqrt(D * t))) - erfc((1 + y + 2 * i) / (2 * np.sqrt(D * t)))
    return sol


# 定义画图函数
def plot_results(times):
    '''
    input times, 
    return C(y)
    '''
    # 绘制 y 为横坐标，浓度为纵坐标的图像
    y=np.linspace(0,1,100)
    plt.figure(figsize=(8, 6))
    for t in times:
        plt.plot(y, analytical_solution(y, t), label=f't={t}')
    plt.xlabel('y')
    plt.ylabel('Concentration')
    plt.legend()
    plt.title('C(y) by analytical method')
    plt.show()


@njit #accelerate
def explicit_finite_difference_result(t, dx, dt, N, D=1.0, num_frames=100):
    '''
    N(int):interval's number
    t=total time
    N (int): grid size
    dt (float): time increment
    dx (float): width of cell in grid
    D (int): diffusion constant
    ''' 
    grid = np.zeros((N, N))
    grid[0,:] = 1
    grid[-1,:] = 0#bound condition
    new_grid = grid.copy()
    num_time_steps = int(t/dt)
    frame_times = np.round(np.linspace(0, num_time_steps - 1, num_frames)).astype(np.int32)
    all_frames = np.zeros((num_frames, N, N))
    index = 0
    for t in range(num_time_steps):

        for i in range(1, N-1):
            for j in range(N):
                new_grid[i, j] = grid[i, j] + dt*D/(dx**2) * (grid[i+1, j] + grid[i-1, j] + grid[i, (j+1)%N] + grid[i, (j-1)%N] - 4*grid[i, j])

        grid[:] = new_grid

        if t in frame_times:
            all_frames[index] = np.copy(grid)
            index += 1
    return  all_frames
    #return grid, all_frames

def ex_plot_result(t_max, dx, dt, N, times, num_frames=100):
    '''
    times = [0, 0.001, 0.01, 0.1, 1]# # 定义时间点
    return C(y) by using explicit finite difference formulation
    '''
    y = np.linspace(0, 1, N)  # 定义 y 的范围
    all_frames = explicit_finite_difference_result(t_max, dx, dt, N, num_frames=100)
    frame_index = [round(t / t_max * (num_frames - 1)) for t in times]
    print(frame_index)
    plt.figure(figsize=(8, 6))
    for t_idx in frame_index:
        t_value = times[frame_index.index(t_idx)] 
    # 提取 c(y) 数据（取中间列）
        reverse_cy = all_frames[t_idx, :, N//2 ]
        cy=reverse_cy[::-1]
        plt.plot(y, cy, label=f't={t_value:.2f}')

    plt.xlabel('y')
    plt.ylabel('Concentration c(y)')
    plt.xticks([-0.1, 1, 1])  # 设定横坐标刻度
    plt.title('explicit finite difference formulation C(y) ')
    plt.legend()
    #plt.grid()
    plt.show()

def plot_2d_concentration(times, t_max, dx, dt, N, D=1.0, num_frames=100):
    '''
    return the 2D heat map by using  explicit finite difference method
    '''
    # 计算所有帧
    all_frames = explicit_finite_difference_result(t_max, dx, dt, N, D, num_frames)

    # 提取指定时间点的帧
    frame_indices = [round(t / t_max * (num_frames - 1)) for t in times]
    frames = [all_frames[idx] for idx in frame_indices]

    # 创建画布
    plt.figure(figsize=(15, 10))
      # 绘制每个时间点的 2D 浓度分布
    for i, (t, frame) in enumerate(zip(times, frames)):
        plt.subplot(2, 3, i + 1)  # 2 行 3 列的子图布局
        plt.imshow(frame, extent=[0, 1, 0, 1], origin='lower', cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label='Concentration')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f't = {t:.3f}')

    plt.tight_layout()  # 自动调整子图间距
    plt.show()

def create_animation(t_max, dx, dt, N, D=1.0, num_frames=100):
    '''
    创建扩散方程的动画
    '''
    # 计算所有帧
    all_frames = explicit_finite_difference_result(t_max, dx, dt, N, D, num_frames)

    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(all_frames[0], extent=[0, 1, 0, 1], origin='lower', cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im, label='Concentration')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Time Dependent Diffusion Equation')
    # 更新函数，用于每一帧
    def update(frame):
        im.set_data(all_frames[frame])
        ax.set_title(f'Time = {frame * t_max / (num_frames - 1):.3f}')
        return im,

    # 创建动画
    ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)

    # 显示动画
    plt.show()




# show the result
D=1
t_max = 1
dx = 1 / 100
dt = 0.25 * dx**2 / (4 * D)
N = 100
times = [0.001,0.01,0.1, 1]
##
# show the analytical C(y)
#plot_results(times)
##explicit finite difference C(y)
#ex_plot_result(t_max, dx, dt, N, times, num_frames=100)
#plot_2d_concentration(times, t_max, dx, dt, N, D=1.0, num_frames=100)
ani=create_animation(t_max, dx, dt, N, D=1.0, num_frames=100)
ani.save(filename="/Users/yuzongyao/Downloads/出国/SFM-STUDY/scinectific computing/hm/diffusion.gif", writer="imagemagick")



    


