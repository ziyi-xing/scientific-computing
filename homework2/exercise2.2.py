import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
import os ##save gif


class MonteCarloDLA:
    def __init__(self, grid_size, ps, max_iter):
        """
        Initialize the Monte Carlo DLA model.
        初始化蒙特卡洛DLA模型。
        :param grid_size: Grid size (height, width).
                         网格大小 (height, width)
        :param ps: Sticking probability.
                  粘附概率
        :param max_iter: Maximum number of iterations.
                        最大迭代次数
        """
        self.grid_size = grid_size
        self.ps = ps
        self.max_iter = max_iter

        # Initialize the grid
        # 初始化网格
        self.grid = np.zeros(grid_size, dtype=int)  # 0 represents an empty point, 1 represents a cluster point
                                                    # 0 表示空点，1 表示团簇点
        self.history = []  # Record the grid state after each cluster change
                           # 记录每次团簇变化后的网格状态

        # Set the initial cluster point (middle of the bottom)
        # 设置初始团簇点（底部中间）
        self.seed_position = (grid_size[0] - 1, grid_size[1] // 2)
        self.grid[self.seed_position] = 1
        self.history.append(self.grid.copy())  # Record the initial state
                                               # 记录初始状态

    def MonteCarlo(self):
        """
        Run the Monte Carlo simulation.
        运行蒙特卡洛模拟。
        """
        self.grid = run_montecarlo(self.grid, self.grid_size, self.ps, self.max_iter, self.history)

    def animate(self):
        """
        Generate an animation.
        生成动画。
        """
        fig, ax = plt.subplots()
        flipped_data = self.history[0][::-1, :] ###要改坐标轴
        img = ax.imshow(flipped_data, cmap='viridis', vmin=0, vmax=1)

        #plt.colorbar(img, label='MC Cluster')
        ax.set_ylim([0,100])

        def update(frame):
            img.set_data(self.history[frame][::-1, :])
            ax.set_title(f" ps={ps}, Step {frame}")
            return img,

        ani = FuncAnimation(fig, update, frames=len(self.history), interval=50, blit=True)
        plt.show()
        return ani
    


@njit
def is_neighbor_of_cluster(grid, x, y):
    """
    Check if the point (x, y) is a neighbor of the cluster.
    检查点 (x, y) 是否是团簇的邻居。
    :param grid: The grid.
                 网格
    :param x: Row coordinate.
              行坐标
    :param y: Column coordinate.
              列坐标
    :return: True if it is a neighbor of the cluster, otherwise False.
             如果是团簇的邻居则返回 True,否则返回 False
    """
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
            if grid[nx, ny] == 1:
                return True
    return False


@njit
def run_montecarlo(grid, grid_size, ps, max_iter, history):
    """
    Run the Monte Carlo simulation, accelerated by Numba.
    运行蒙特卡洛模拟，使用 Numba 加速。
    """
    for _ in range(max_iter):
        x, y = 0, np.random.randint(0, grid_size[1])  # Release a particle randomly at the top
                                                      # 在顶部随机释放一个粒子

        while True:
            # Random walk
            # 随机游走
            direction = np.random.randint(4)
            if direction == 0:  # up
                x_new, y_new = x - 1, y
            elif direction == 1:  # down
                x_new, y_new = x + 1, y
            elif direction == 2:  # left
                x_new, y_new = x, y - 1
            else:  # right
                x_new, y_new = x, y + 1

            # Check boundary conditions
            # 检查边界条件
            if x_new < 0:  # Exceed the top boundary
                           # 超出上边界
                break  # Stop the simulation if the cluster touches the top boundary
                       # 如果团簇接触到上边界，则停止模拟
            if x_new >= grid_size[0]:  # Exceed the bottom boundary
                                       # 超出下边界
                break  # Remove the particle and release a new one
                       # 移除粒子并释放新的粒子
            if y_new < 0:  # Exceed the left boundary
                           # 超出左边界
                y_new = grid_size[1] - 1
            elif y_new >= grid_size[1]:  # Exceed the right boundary
                                         # 超出右边界
                y_new = 0
            # 检查是否移动到团簇内部
            if grid[x_new, y_new] == 1:
                    continue  # 不能移动到团簇内部，重新选择方向


            # Check if the particle reaches the neighborhood of the cluster
            # 检查是否到达团簇的邻域
            if is_neighbor_of_cluster(grid, x_new, y_new) and (grid[x_new, y_new] != 1):
                if np.random.rand() < ps:  # Stick with probability ps
                                           # 以概率 ps 粘附
                    grid[x_new, y_new] = 1  # Join the cluster
                                            # 加入团簇
                    history.append(grid.copy())  # Record the current state
                                                # 记录当前状态
                    if x_new == 0:  # If the cluster touches the top boundary
                                    # 如果团簇接触到上边界
                        return grid  # Stop the simulation
                                     # 停止模拟
                    break
                else:
                    x, y = x_new, y_new  # Continue walking
                                          # 继续游走
            else:
                x, y = x_new, y_new  # Continue walking
                                      # 继续游走
    return grid


# Test code
# 测试代码

 # 测试代码
# if __name__ == "__main__":
#     # 初始化蒙特卡洛DLA模型
#     dla = MonteCarloDLA(grid_size=(100, 100), ps=1, max_iter=80000)

#     # 运行蒙特卡洛模拟
#     dla.MonteCarlo()

#     # 生成动画并获取 ani 对象
#     ani = dla.animate()

#     # 保存动画为 GIF
#     figures_dir = os.path.join(os.path.dirname(__file__), "figures2.2")  # 保存结果的文件夹
#     os.makedirs(figures_dir, exist_ok=True)  # 如果目录不存在则创建
#     gif_path = os.path.join(figures_dir, "dla_MCanimation.gif")  # GIF 文件路径
#     ani.save(gif_path, writer="pillow", fps=15)  # 保存为 GIF，fps 控制帧率
#     print(f"Animation saved to {gif_path}")  # 打印 GIF 保存的路径

#     # 保存最后一帧的图像
#     final_frame = dla.history[-1][::-1, :]  # 获取最后一帧
#     final_image_path = os.path.join(figures_dir, "MC_final_frame.png")  # 最后一帧图像路径
#     plt.imshow(final_frame, cmap='viridis', vmin=0, vmax=1, origin='lower')  # 显示最后一帧
#     plt.title("Final Frame")  # 设置标题
#     plt.colorbar(label='Cluster')  # 添加颜色条
#     plt.savefig(final_image_path)  # 保存最后一帧图像
#     plt.close()  # 关闭图像窗口
#     print(f"Final frame saved to {final_image_path}")  # 打印最后一帧保存的路径                                                
#     print(f"Animation saved to {gif_path}")  # Print the path where the GIF is saved
#                                              # 打印GIF保存的路径



####when changing ps
# 测试代码

###3需要解决的东西
# 1.生成不同的ps，生成图片
# 2.保存方面
# 3.导入git
# 4.删除测试的文件
# Test code
# if __name__ == "__main__":
#     # 定义不同的 ps 值
#     ps_values = [0.1, 0.5, 0.7, 1]

#     # 创建保存结果的文件夹
#     figures_dir = os.path.join(os.path.dirname(__file__), "figures2.2")
#     os.makedirs(figures_dir, exist_ok=True)  # 如果目录不存在则创建

#     for ps in ps_values:
#         # 初始化蒙特卡洛DLA模型
#         dla = MonteCarloDLA(grid_size=(100, 100), ps=ps, max_iter=70500)

#         # 运行蒙特卡洛模拟
#         #step_when_top_reached, final_grid = 
#         dla.MonteCarlo()

#         # 生成动画并获取 ani 对象
#         ani = dla.animate()

#         # 保存动画为 GIF
#         gif_path = os.path.join(figures_dir, f"dla_MCanimation_ps_{ps}.gif")  # 动态生成文件名
#         ani.save(gif_path, writer="pillow", fps=15)  # 保存为 GIF，fps 控制帧率
#         print(f"Animation saved to {gif_path}")  # 打印 GIF 保存的路径

#         # 保存最后一帧的图像
#         final_frame = dla.history[-1][::-1, :]  # 获取最后一帧
#         final_image_path = os.path.join(figures_dir, f"MC_final_frame_ps_{ps}.png")  # 动态生成文件名
#         plt.imshow(final_frame, cmap='viridis', vmin=0, vmax=1, origin='lower')
#           # 显示最后一帧
#         step_when_top_reached=len(dla.history)  
#         plt.title(f"ps={ps}, Step={step_when_top_reached}")  # 设置标题，标注 ps 和步数
#         #plt.colorbar(label='Cluster')  # 添加颜色条
#         plt.savefig(final_image_path)  # 保存最后一帧图像
#         plt.close()  # 关闭图像窗口
#         print(f"Final frame saved to {final_image_path}")  # 打印最后一帧保存的路径

if __name__ == "__main__":
    # 定义不同的 ps 值
    ps_values = [0.1, 0.5, 0.7, 1]

    # 创建保存结果的文件夹
    figures_dir = os.path.dirname(os.path.join(os.path.dirname(__file__), "figures2.2"))
    os.makedirs(figures_dir, exist_ok=True)  # 如果目录不存在则创建

    # 创建一个空的2x2图像
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2x2网格

    for i, ps in enumerate(ps_values):
        # 初始化蒙特卡洛DLA模型
        dla = MonteCarloDLA(grid_size=(100, 100), ps=ps, max_iter=50000)

        # 运行蒙特卡洛模拟
        dla.MonteCarlo()

        # 获取最后一帧
        final_frame = dla.history[-1][::-1, :]  # 获取最后一帧

        # 获取对应的位置
        ax = axes[i // 2, i % 2]

        # 绘制最后一帧的图像
        ax.imshow(final_frame, cmap='viridis', vmin=0, vmax=1, origin='lower')
        step_when_top_reached = len(dla.history)
        ax.set_title(f"ps={ps}, Step={step_when_top_reached}")  # 设置标题
        # ax.axis('off')  # 关闭坐标轴

        # 生成动画并获取 ani 对象
        ani = dla.animate()

        # 保存动画为 GIF
        gif_path = os.path.join(figures_dir, f"dla_MCanimation_ps_{ps}.gif")  # 动态生成文件名
        ani.save(gif_path, writer="pillow", fps=15)  # 保存为 GIF，fps 控制帧率
        print(f"Animation saved to {gif_path}")  # 打印 GIF 保存的路径

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存合并后的图像
    combined_image_path = os.path.join(figures_dir, "combined_final_frames.png")
    plt.savefig(combined_image_path)
    plt.close()  # 关闭图像窗口
    print(f"Combined final frames saved to {combined_image_path}")  # 打印保存路径
