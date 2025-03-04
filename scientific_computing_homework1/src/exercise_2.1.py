import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
from numba import jit

# SOR求解器
@jit(nopython=True)
def solve_diffusion(grid, sinks, omega=1.7, tol=1e-5, max_iter=10000):
    """
    使用SOR方法求解扩散方程。
    :param grid: 网格场
    :param sinks: 固定点（粒子聚集点）
    :param omega: SOR松弛因子
    :param tol: 收敛容差
    :param max_iter: 最大迭代次数
    :return: 更新后的网格场和迭代次数
    """
    ny, nx = grid.shape
    residual = tol + 1  # 确保至少一次迭代

    for iteration in range(max_iter):
        residual = 0.0

        for i in range(1, ny - 1):
            for j in range(nx):
                if sinks[i, j]:
                    continue  # 跳过固定点

                # 周期性边界条件
                left = grid[i, (j - 1) % nx]
                right = grid[i, (j + 1) % nx]
                up = grid[i - 1, j]
                down = grid[i + 1, j]

                # SOR更新
                old_value = grid[i, j]
                new_value = (1 - omega) * old_value + omega * (up + down + left + right) / 4.0
                new_value = max(0.0, new_value)
                residual = max(residual, abs(new_value - old_value))
                grid[i, j] = new_value

        if residual < tol:
            return grid, iteration + 1

    return grid, max_iter

def get_growth_candidates(cluster_field):
    """
    获取生长候选点。
    :param cluster_field: 粒子聚集场
    :return: 候选点的坐标列表
    """
    east = np.roll(cluster_field, shift=-1, axis=1)
    west = np.roll(cluster_field, shift=1, axis=1)
    north = np.roll(cluster_field, shift=1, axis=0)
    south = np.roll(cluster_field, shift=-1, axis=0)

    # 排除无效边界
    north[0, :] = 0  # 顶部边界无效
    south[-1, :] = 0  # 底部边界无效

    # 候选点条件：至少有一个邻居是已聚集的粒子，且自身未被占据
    neighbor_occupied = (north + south + east + west) > 0
    candidates = (cluster_field == 0) & neighbor_occupied

    return np.argwhere(candidates)

def choose_growth_candidate(candidates, nutrient_field, eta):
    """
    根据营养浓度选择生长点。
    :param candidates: 候选点坐标列表
    :param nutrient_field: 营养浓度场
    :param eta: 控制生长概率的参数
    :return: 选择的生长点坐标
    """
    if len(candidates) == 0:
        return None

    # 计算候选点的生长概率
    nutrient_values = nutrient_field[candidates[:, 0], candidates[:, 1]] ** eta
    probabilities = nutrient_values / nutrient_values.sum()

    # 随机选择一个生长点
    chosen_index = candidates[np.random.choice(len(candidates), p=probabilities)]
    return chosen_index

class DiffusionLimitedAggregation:
    """
    DLA模型类，用于模拟扩散限制聚集过程。
    """
    def __init__(self, grid_size: tuple, eta: float):
        """
        初始化DLA模型。
        :param grid_size: 网格大小 (height, width)
        :param eta: 控制生长概率与营养浓度关系的参数
        """
        self.grid_size = grid_size
        self.eta = eta
        self.nutrient_field = np.zeros(grid_size)  # 营养浓度场
        self.cluster_field = np.zeros_like(self.nutrient_field)  # 粒子聚集场
        self.nutrient_field[0, :] = 1.0  # 顶部边界设置为营养源

        # 在网格底部中心设置初始种子
        seed_position = grid_size[1] // 2
        self.cluster_field[-1, seed_position] = 1
        self.nutrient_field[-1, seed_position] = 0.0  # 种子处设置为固定点

        self.termination_flag = False  # 终止标志
        self.termination_step = -1  # 终止步数
        self.history = []  # 记录每次迭代的网格状态

    def update_nutrient_field(self, omega=1.7, tol=1e-5, max_iter=10000):
        """
        更新营养浓度场。
        :param omega: SOR松弛因子
        :param tol: 收敛容差
        :param max_iter: 最大迭代次数
        :return: 迭代次数
        """
        sinks = self.cluster_field == 1  # 固定点
        self.nutrient_field, iter_count = solve_diffusion(self.nutrient_field, sinks, omega=omega, tol=tol, max_iter=max_iter)
        return iter_count

    def grow(self, growth_steps, plot_interval=100, omega=1.7, tol=1e-5, max_iter=10000):
        """
        运行DLA生长过程。
        :param growth_steps: 最大生长步数
        :param plot_interval: 绘制间隔（若为0则不绘制）
        :param omega: SOR松弛因子
        :param tol: 收敛容差
        :param max_iter: 最大迭代次数
        """
        for step in range(growth_steps):
            if self.termination_flag:
                self.termination_step = step
                print(f"Termination at step {step} with η = {self.eta}")
                break

            self.update_nutrient_field(omega=omega, tol=tol, max_iter=max_iter)
            candidates = get_growth_candidates(self.cluster_field)
            chosen_index = choose_growth_candidate(candidates, self.nutrient_field, self.eta)

            if chosen_index is not None:
                self.cluster_field[chosen_index[0], chosen_index[1]] = 1
                self.nutrient_field[chosen_index[0], chosen_index[1]] = 0  # 设置为固定点

                # 检查是否到达顶部边界
                if chosen_index[0] == 0:
                    self.termination_flag = True

            # 记录当前状态
            self.history.append((self.nutrient_field.copy(), self.cluster_field.copy()))

            if plot_interval > 0 and (step + 1) % plot_interval == 0:
                print(f"Step {step + 1} with η = {self.eta}")

        if self.termination_step < 0:
            self.termination_step = growth_steps

def update_animation(frame, img, history, dla):
    """
    更新动画帧。
    :param frame: 当前帧数
    :param img: 图像对象
    :param history: 历史记录
    :param dla: DLA 模型实例
    :return: 更新后的图像对象
    """
    nutrient_field, cluster_field = history[frame]
    image = nutrient_field.copy()
    image[cluster_field == 1] = 1  # 将团簇点的值设置为1
    img.set_data(image)

    # 更新标题，包含 eta 值
    ax.set_title(f"Step {frame} with η = {dla.eta}")  # 添加 eta 值
    fig.canvas.draw_idle()  # 强制刷新图像，确保标题更新

    # 触顶时停止动画但不关闭窗口
    if frame >= dla.termination_step - 1:
        print("Animation stopped at step:", frame)
        ani.event_source.stop()  # 只停止动画，不关闭窗口

    return img,

def calculate_fractal_dimension(cluster_field):
    """
    使用盒计数法计算分形维度。
    :param cluster_field: 粒子聚集场
    :return: 分形维度
    """
    cluster_points = np.argwhere(cluster_field == 1)
    if len(cluster_points) == 0:
        return 0.0

    # 定义盒子大小范围
    box_sizes = np.logspace(0, np.log10(min(cluster_field.shape)), base=10, num=20)
    box_sizes = np.unique(np.floor(box_sizes)).astype(int)
    box_sizes = box_sizes[box_sizes > 0]

    # 计算每个盒子大小下的盒子数量
    box_counts = []
    for box_size in box_sizes:
        bins = [np.arange(0, cluster_field.shape[i] + box_size, box_size) for i in range(2)]
        hist, _, _ = np.histogram2d(cluster_points[:, 0], cluster_points[:, 1], bins=bins)
        box_counts.append(np.sum(hist > 0))

    # 拟合分形维度
    if len(box_sizes) < 2:
        return 0.0

    coeffs = np.polyfit(np.log(1 / box_sizes), np.log(box_counts), 1)
    return coeffs[0]

def study_eta_impact(eta_values, grid_size=(100, 100), growth_steps=5000, omega=1.7):
    """
    研究 eta 参数对 DLA 模型的影响。
    :param eta_values: eta 值列表
    :param grid_size: 网格大小
    :param growth_steps: 最大生长步数
    :param omega: SOR松弛因子
    """
    results = []
    fractal_dimensions = []
    for eta in eta_values:
        dla = DiffusionLimitedAggregation(grid_size, eta)
        dla.grow(growth_steps, plot_interval=0, omega=omega)
        results.append((eta, dla.termination_step, dla.cluster_field, dla.nutrient_field))

        # 计算分形维度
        fractal_dim = calculate_fractal_dimension(dla.cluster_field)
        fractal_dimensions.append(fractal_dim)
        print(f"η = {eta}, Fractal Dimension = {fractal_dim:.4f}")

    # 绘制不同 eta 值下的团簇形态
    plt.figure(figsize=(10, 10))  # 调整画布大小以适应 2x2 布局

    for i, (eta, step, cluster_field, nutrient_field) in enumerate(results):
        plt.subplot(2, 2, i + 1)  # 2 行 2 列，第 i+1 个子图
        
        # 结合 nutrient_field 和 cluster_field 绘制渐变热力图
        image = nutrient_field.copy()
        image[cluster_field == 1] = 1  # 将团簇点的值设置为 1
        plt.imshow(image, cmap="viridis", vmin=0, vmax=1)
        
        # 设置标题，只显示 eta 值和步数
        plt.title(f"η = {eta}, Steps = {step}")

    # 调整子图间距
    plt.tight_layout()

    # 保存图片到 figures 文件夹
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")  # 获取 figures 文件夹路径
    os.makedirs(figures_dir, exist_ok=True)  # 如果文件夹不存在，则创建
    plt.savefig(os.path.join(figures_dir, "eta_impact_grid.png"))  # 保存图片
    plt.show()

    # 绘制 eta 与分形维度的关系图
    plt.figure()
    plt.plot(eta_values, fractal_dimensions, marker='o')
    plt.xlabel('Eta (η)')
    plt.ylabel('Fractal Dimension')
    plt.title('Impact of Eta on Fractal Dimension')
    
    # 保存图片到 figures 文件夹
    plt.savefig(os.path.join(figures_dir, "eta_fractal_dimension.png"))  # 保存图片
    plt.show()

def study_eta_omega_impact(eta_values, omega_values, grid_size=(100, 100), growth_steps=5000):
    """
    研究不同 eta 和 omega 值对 SOR 迭代次数的影响。
    :param eta_values: 要测试的 eta 值列表
    :param omega_values: 要测试的 omega 值列表
    :param grid_size: 网格大小
    :param growth_steps: 最大生长步数
    """
    results = []
    for eta in eta_values:
        for omega in omega_values:
            dla = DiffusionLimitedAggregation(grid_size, eta)
            total_iterations = 0  # 记录总的 SOR 迭代次数
            for step in range(growth_steps):
                if dla.termination_flag:
                    break

                # 更新营养场并记录迭代次数
                iter_count = dla.update_nutrient_field(omega=omega)
                total_iterations += iter_count

                # 生长过程
                candidates = get_growth_candidates(dla.cluster_field)
                chosen_index = choose_growth_candidate(candidates, dla.nutrient_field, eta)
                if chosen_index is not None:
                    dla.cluster_field[chosen_index[0], chosen_index[1]] = 1
                    dla.nutrient_field[chosen_index[0], chosen_index[1]] = 0  # 设置为固定点

                    # 检查是否到达顶部边界
                    if chosen_index[0] == 0:
                        dla.termination_flag = True

            # 记录当前 eta 和 omega 对应的平均迭代次数
            average_iterations = total_iterations / (step + 1)  # 平均每次更新营养场的迭代次数
            results.append((eta, omega, average_iterations))

    # 输出每个 eta 和 omega 组合的平均 SOR 迭代次数
    print("Eta and Omega Impact on SOR Iterations:")
    print("Eta (η) | Omega (ω) | Average SOR Iterations")
    print("--------------------------------------------")
    for result in results:
        eta, omega, avg_iter = result
        print(f"{eta:7.2f} | {omega:9.2f} | {avg_iter:20.2f}")

    # 将结果转换为二维数组以便绘图
    eta_grid, omega_grid = np.meshgrid(eta_values, omega_values)  # 交换 eta 和 omega 的顺序
    iteration_grid = np.zeros_like(eta_grid, dtype=float)

    for result in results:
        eta, omega, avg_iter = result
        eta_index = eta_values.index(eta)
        omega_index = omega_values.index(omega)
        iteration_grid[omega_index, eta_index] = avg_iter  # 注意索引顺序

    # 绘制等高线图
    plt.figure(figsize=(10, 6))
    plt.contourf(eta_grid, omega_grid, iteration_grid, levels=50, cmap='viridis')
    plt.colorbar(label='Average SOR Iterations')
    plt.xlabel('Eta (η)')
    plt.ylabel('Omega (ω)')
    plt.title('Impact of Eta and Omega on SOR Iterations (Contour Plot)')
    # 保存图片到 figures 文件夹
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")  # 获取 figures 文件夹路径
    os.makedirs(figures_dir, exist_ok=True)  # 如果文件夹不存在，则创建
    plt.savefig(os.path.join(figures_dir, "eta_omega_impact.png"))  # 保存图片
    plt.show()
if __name__ == "__main__":
    # 参数设置
    grid_size = (100, 100)  # 网格大小
    eta = 1.0  # 控制生长概率的参数
    growth_steps = 5000  # 最大生长步数
    plot_interval = 100  # 绘制间隔

    # 初始化DLA模型
    dla = DiffusionLimitedAggregation(grid_size, eta)
    dla.grow(growth_steps, plot_interval)

    # 创建动画
    fig, ax = plt.subplots()
    nutrient_field, cluster_field = dla.history[0]
    image = nutrient_field.copy()
    image[cluster_field == 1] = 1  # 让团簇点显示
    img = ax.imshow(image, cmap="viridis", vmin=0, vmax=1)

    # 使用 FuncAnimation
    ani = FuncAnimation(fig, update_animation, fargs=(img, dla.history, dla),
                        frames=len(dla.history), interval=50, blit=False)

    plt.colorbar(img, label="Nutrient Concentration / Cluster")

    # 保存动画为 GIF 文件
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")  # 获取 figures 文件夹路径
    os.makedirs(figures_dir, exist_ok=True)  # 如果文件夹不存在，则创建
    gif_path = os.path.join(figures_dir, "dla_animation.gif")  # GIF 文件路径
    ani.save(gif_path, writer="pillow", fps=15)  # 保存为 GIF 文件，fps 控制帧率
    print(f"Animation saved to {gif_path}")

    plt.show()

    # 研究 eta 参数的影响
    eta_values = [0.5, 1.0, 1.5, 2.0]
    study_eta_impact(eta_values)

    # 研究 eta 和 omega 的影响
    omega_values = [1.0, 1.2, 1.5, 1.7, 1.9]
    study_eta_omega_impact(eta_values, omega_values, grid_size, growth_steps)