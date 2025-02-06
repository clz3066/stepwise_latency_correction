import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import os 

mm  = 1/25.4 # inch 和 毫米的转换
mat = scio.loadmat('./individual_level/paper_figures/sample.mat')['results']
data = mat[0, 0]['original']
print(data.shape)

def draw_noisy_parallel_lines(rectangle, num_lines, noise_scale):
    # 创建一个图形和坐标轴
    fig, axs = plt.subplots(1, 3, figsize=(120*mm, 20*mm))
    for idx in range(3):
        ax = axs[idx]
        ax.set_facecolor('white')
        
        # 绘制长方形框
        rect = plt.Rectangle((rectangle[0], rectangle[1]), rectangle[2], rectangle[3], fill=None, edgecolor='black', linewidth=3.0)
        ax.add_patch(rect)

        # 计算每条线的 y 坐标
        y_start = rectangle[1]
        y_end = rectangle[1] + rectangle[3]
        x_values = np.linspace(rectangle[0], rectangle[0] + rectangle[2], 100)

        for i in range(num_lines):
            # 均匀分布在长方形高度上，确保不会超出边界
            y_coord = y_start + (i / (num_lines - 1)) * rectangle[3]  
            noise = np.random.normal(0, noise_scale, x_values.shape)  # 生成噪声
            noisy_y = y_coord + noise  # 添加噪声

            # 限制噪声线条在长方形内
            noisy_y = np.clip(noisy_y, y_start, y_end)

            ax.plot(x_values, noisy_y - 0.1, color='#334c81', linewidth=1.0)

        # 设置坐标轴范围
        ax.set_xlim(rectangle[0], rectangle[0] + rectangle[2])
        ax.set_ylim(rectangle[1], rectangle[1] + rectangle[3])
        
        # 设置比例相同
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')  # 关闭坐标轴

    plt.show()
    fig.savefig('./individual_level/paper_figures/fig3/trials1.svg', bbox_inches='tight', transparent=True)
    plt.close()

# 示例参数: (x, y, width, height), 线条数量, 噪声规模
draw_noisy_parallel_lines((1, 1, 4, 3), num_lines=8, noise_scale=0.1)
