import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# --- 1. 数据准备 (不变) ---
intervals = ['3 out of 3', '2 out of 3', '1 out of 3', '0 out of 3']
model_qwen = 'Qwen2.5-7B-Instruct' # 稍微补全名字以匹配风格
model_qwe3 = 'Qwe3-4B-Instruct'

kendall_single_qwen = [90.39, 88.29, 72.90, 55.42]
kendall_single_qwe3 = [86.76, 84.36, 70.35, 40.70]

kendall_multi_qwen = [82.12, 78.46, 65.75, 60.45]
kendall_multi_qwe3 = [79.50, 75.64, 62.50, 55.08]

# --- 2. 样式与画布设置 ---
# 保持完全一致的风格
plt.style.use('seaborn-v0_8-darkgrid')

# 【关键修正1】 画布大小严格一致 (12, 11.3)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 11.3))

bar_width = 0.30
x = np.arange(len(intervals))

# 【关键修正2】 字体大小严格对齐图1
# 图1配置: Title=24, Label=24, Tick=22, Legend=22
FONT_TITLE = 24
FONT_LABEL = 24
FONT_TICK = 22
FONT_LEGEND = 22
FONT_BAR_VAL = 20  # 柱状图特有数值标签，适配大图

# Y轴刻度配置
Y_TICKS = np.arange(30, 101, 20) 
Y_LIM = (30, 100)

# --- 3. 绘制子图 1 (Single Hop) ---
rects1_qwen = ax1.bar(x - bar_width/2, kendall_single_qwen, bar_width, 
                      label=model_qwen, color='#4C72B0')
rects1_qwe3 = ax1.bar(x + bar_width/2, kendall_single_qwe3, bar_width, 
                      label=model_qwe3, color='#55A868')

ax1.set_ylabel('Kendall-tau Score', fontsize=FONT_LABEL)
ax1.set_title('General QA Verification: Kendall-tau Score by Interval', fontsize=FONT_TITLE, weight='bold')

# 刻度设置
ax1.set_xticks(x)
ax1.set_xticklabels(intervals, fontsize=FONT_TICK)
ax1.tick_params(axis='y', labelsize=FONT_TICK)

# 图例与范围
ax1.legend(fontsize=FONT_LEGEND, loc='upper right')
ax1.set_yticks(Y_TICKS) 
ax1.set_ylim(Y_LIM)

# 数值标签
ax1.bar_label(rects1_qwen, padding=3, fmt='%.2f', fontsize=FONT_BAR_VAL)
ax1.bar_label(rects1_qwe3, padding=3, fmt='%.2f', fontsize=FONT_BAR_VAL)

# --- 4. 绘制子图 2 (Multi Hop) ---
rects2_qwen = ax2.bar(x - bar_width/2, kendall_multi_qwen, bar_width, 
                      label=model_qwen, color='#4C72B0')
rects2_qwe3 = ax2.bar(x + bar_width/2, kendall_multi_qwe3, bar_width, 
                      label=model_qwe3, color='#55A868')

ax2.set_ylabel('Kendall-tau Score', fontsize=FONT_LABEL)
ax2.set_title('Multi-hop QA Verification: Kendall-tau Score by Interval', fontsize=FONT_TITLE, weight='bold')

# 刻度设置
ax2.set_xticks(x)
ax2.set_xticklabels(intervals, fontsize=FONT_TICK)
ax2.tick_params(axis='y', labelsize=FONT_TICK)

# 图例与范围
ax2.legend(fontsize=FONT_LEGEND, loc='upper right')
ax2.set_yticks(Y_TICKS)
ax2.set_ylim(Y_LIM)

# 数值标签
ax2.bar_label(rects2_qwen, padding=3, fmt='%.2f', fontsize=FONT_BAR_VAL)
ax2.bar_label(rects2_qwe3, padding=3, fmt='%.2f', fontsize=FONT_BAR_VAL)

# --- 5. 输出设置 (最关键的Margin修正) ---

# 【关键修正3】 使用与图1完全一致的 tight_layout 且不带参数
# 图1代码: plt.tight_layout() -> 这里的行为必须一致
plt.tight_layout() 

# 【关键修正4】 移除 bbox_inches='tight'，并统一 DPI 为 800
# 图1代码: plt.savefig('linechart.png', dpi=800)
plt.savefig('kendall_comparison.png', dpi=800)

plt.show()