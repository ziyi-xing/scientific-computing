import numpy as np
import matplotlib.pyplot as plt

# 设置热传导方程参数
L = 1.0    # 空间长度
Nx = 100   # 空间网格数
alpha = 0.01  # 热扩散系数
dt = 0.001    # 时间步长
total_time = 0.1  # 总时间

# 初始化网格和温度分布
x = np.linspace(0, L, Nx + 1)
u = np.sin(np.pi * x)

# 时间迭代（显式差分法）
for _ in range(int(total_time / dt)):
    u[1:-1] += alpha * dt / (L/Nx)**2 * (u[2:] - 2*u[1:-1] + u[:-2])

# 结果可视化
plt.plot(x, u, label="Numerical Solution")
plt.title("1D Heat Equation")
plt.xlabel("Position")
plt.ylabel("Temperature")
plt.legend()
plt.show()