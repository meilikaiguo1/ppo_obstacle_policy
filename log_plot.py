import numpy as np
import matplotlib.pyplot as plt


def logistic(x, growth_rate, mid_point):
    z = -growth_rate * (x - mid_point)
    return np.where(z > 700, 0, 1 / (1 + np.exp(z)))


# 参数设置
growth_rate = 0.032
mid_point = 75

# print(logistic(2000, growth_rate, mid_point) - 1.0)
# print(logistic(1000, growth_rate, mid_point) - 1.0)
# 横坐标范围
x = np.linspace(-200, 200, 10)
y = logistic(x, growth_rate, mid_point)
print(logistic(0, growth_rate, mid_point))

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(x, y, linewidth=2, label=f'growth_rate={growth_rate}, mid_point={mid_point}')
plt.axvline(mid_point, linestyle='--', linewidth=1.5, label=f'mid_point={mid_point}')
plt.xlabel("x")
plt.ylabel("logistic(x)")
plt.title("Logistic Function")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()