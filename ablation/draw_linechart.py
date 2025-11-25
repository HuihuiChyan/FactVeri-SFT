import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import matplotlib

# --- 字体配置 ---
TITLE_FONTSIZE = 24    
LABEL_FONTSIZE = 24    
TICK_FONTSIZE = 22     
LEGEND_FONTSIZE = 22   

# 应用 'seaborn-darkgrid' 样式
plt.style.use('seaborn-v0_8-darkgrid')

# 设置中文字体 (如果需要)
zhfont = matplotlib.font_manager.FontProperties(fname="SourceHanSansSC-Bold.otf") 

# 1. 从图片中转录数据 (***已更新：加入了 D-Verifier 的数据***)
data = {
    'Model': [
        # 0.5B
        'Qwen2.5-0.5B-Instruct', 'Qwen2.5-0.5B-Instruct', 'Qwen2.5-0.5B-Instruct', 'Qwen2.5-0.5B-Instruct',
        # 1.5B
        'Qwen2.5-1.5B-Instruct', 'Qwen2.5-1.5B-Instruct', 'Qwen2.5-1.5B-Instruct', 'Qwen2.5-1.5B-Instruct',
        # 3B
        'Qwen2.5-3B-Instruct', 'Qwen2.5-3B-Instruct', 'Qwen2.5-3B-Instruct', 'Qwen2.5-3B-Instruct',
        # 7B
        'Qwen2.5-7B-Instruct', 'Qwen2.5-7B-Instruct', 'Qwen2.5-7B-Instruct', 'Qwen2.5-7B-Instruct',
        # 14B
        'Qwen2.5-14B-Instruct', 'Qwen2.5-14B-Instruct', 'Qwen2.5-14B-Instruct', 'Qwen2.5-14B-Instruct'
    ],
    'Verifier': [
        # 0.5B
        'G-Verifier', 'AG-Verifier', 'D-Verifier', 'DiVA',
        # 1.5B
        'G-Verifier', 'AG-Verifier', 'D-Verifier', 'DiVA',
        # 3B
        'G-Verifier', 'AG-Verifier', 'D-Verifier', 'DiVA',
        # 7B
        'G-Verifier', 'AG-Verifier', 'D-Verifier', 'DiVA',
        # 14B
        'G-Verifier', 'AG-Verifier', 'D-Verifier', 'DiVA'
    ],
    'TriviaQA_Precision@1': [
        43.55, 40.98, 47.31, 86.02,  # 0.5B
        51.34, 53.23, 52.42, 87.37,  # 1.5B
        61.39, 71.31, 58.33, 88.98,  # 3B
        69.81, 78.98, 74.73, 90.86,  # 7B
        69.69, 82.21, 77.15, 90.65   # 14B
    ],
    'TriviaQA_Kendall': [
        12.01, 12.20, 18.46, 82.62,  # 0.5B
        31.00, 32.08, 29.39, 84.41,  # 1.5B
        45.36, 59.66, 41.40, 84.41,  # 3B
        61.37, 72.69, 62.54, 88.53,  # 7B
        61.21, 80.05, 67.56, 86.52   # 14B
    ],
    'HotpotQA_Precision@1': [
        40.56, 33.74, 40.77, 76.66,  # 0.5B
        42.51, 45.99, 47.74, 80.84,  # 1.5B
        52.89, 59.93, 55.40, 86.06,  # 3B
        63.99, 75.26, 63.07, 83.28,  # 7B
        75.68, 81.88, 71.08, 84.56   # 14B
    ],
    'HotpotQA_Kendall': [
        10.96, 1.02,  8.71,  65.62,  # 0.5B
        22.18, 18.70, 27.06, 70.27,  # 1.5B
        34.71, 42.32, 34.03, 78.63,  # 3B
        41.96, 59.12, 43.32, 77.24,  # 7B
        71.53, 75.15, 56.79, 79.53   # 14B
    ]
}
df = pd.DataFrame(data)

# 2. 提取模型大小作为x轴的数值
def extract_size(model_name):
    match = re.search(r'(\d+(\.\d+)?B)', model_name)
    if match:
        return float(match.group(1).replace('B', ''))
    return None

df['Model_Size'] = df['Model'].apply(extract_size)

# 获取排序后的模型大小和对应的标签
model_sizes = sorted(df['Model_Size'].unique())
model_size_labels = [f"{size}B" for size in model_sizes]

# 定义想要绘制的 Verifier 顺序 (图例也会按这个顺序)
verifiers = ['G-Verifier', 'AG-Verifier', 'D-Verifier', 'DiVA']

# 创建等间距的X轴位置 [0, 1, 2, 3, 4]
x_positions = np.arange(len(model_size_labels))

# *** 颜色配置 (新增了 D-Verifier 的红色) ***
colors = {
    'G-Verifier': '#1f77b4',   # 蓝
    'AG-Verifier': '#ff7f0e',  # 橙
    'D-Verifier': '#d62728',   # 红 (新增)
    'DiVA': '#2ca02c'   # 绿
}

# 3. 创建2x1的子图
fig, axes = plt.subplots(2, 1, figsize=(12, 11.3))
fig.suptitle('Kendall-tau Score vs. Qwen Model Size', fontsize=TITLE_FONTSIZE, y=0.985, weight='bold')

# --- 图 1: TriviaQA - Kendall (上图) ---
ax1 = axes[0]
for verifier in verifiers:
    plot_data = df[df['Verifier'] == verifier].sort_values('Model_Size')
    ax1.plot(x_positions, plot_data['TriviaQA_Kendall'], marker='o', label=verifier, color=colors[verifier], linewidth=2)

ax1.set_xlabel('Model Size', fontsize=LABEL_FONTSIZE)
ax1.set_ylabel('TriviaQA', fontsize=LABEL_FONTSIZE)

ax1.set_xticks(x_positions)
ax1.set_xticklabels(model_size_labels, fontsize=TICK_FONTSIZE)
ax1.tick_params(axis='y', labelsize=TICK_FONTSIZE) 
ax1.legend(fontsize=LEGEND_FONTSIZE)

# --- 图 2: HotpotQA - Kendall (下图) ---
ax2 = axes[1]
for verifier in verifiers:
    plot_data = df[df['Verifier'] == verifier].sort_values('Model_Size')
    ax2.plot(x_positions, plot_data['HotpotQA_Kendall'], marker='o', label=verifier, color=colors[verifier], linewidth=2)

ax2.set_xlabel('Model Size', fontsize=LABEL_FONTSIZE)
ax2.set_ylabel('HotpotQA', fontsize=LABEL_FONTSIZE)

ax2.set_xticks(x_positions)
ax2.set_xticklabels(model_size_labels, fontsize=TICK_FONTSIZE)
ax2.tick_params(axis='y', labelsize=TICK_FONTSIZE) 
ax2.legend(fontsize=LEGEND_FONTSIZE)

# 调整布局并保存
plt.tight_layout() 
plt.savefig('linechart.png', dpi=800)
plt.show()