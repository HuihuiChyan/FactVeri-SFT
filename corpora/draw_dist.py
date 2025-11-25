import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from collections import Counter
import numpy as np

# --- 审美配置区域 (Aesthetics Config) ---
# 1. 全局字体设置：使用更现代的无衬线字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'SimHei'] # 兼顾英文和中文环境

# 2. 高级配色方案 (Hex Codes)
# 这一组颜色更柔和、专业，且区分度高
CUSTOM_COLORS = [
    "#4E79A7", # 深蓝 (Llama-3.1 等主力模型)
    "#F28E2B", # 橙色
    "#E15759", # 红色
    "#76B7B2", # 青色
    "#59A14F", # 绿色
    "#EDC948", # 黄色
    "#B07AA1", # 紫色
    "#FF9DA7", # 粉色
    "#9C755F", # 棕色
    "#BAB0AC", # 灰色
    "#86BCB6", # 浅青
    "#E19D29", # 深黄
    "#4B9ac7", # 亮蓝
]
OTHERS_COLOR = '#EAEAEA' # Others 用非常淡的灰色，降低视觉干扰
OTHERS_LABEL = 'Others'

# --- 业务配置区域 ---
INPUT_FILE = 'concat_verification.jsonl'
THRESHOLD_PERCENT = 5.0 

# 初始化
rank_1_counts = Counter()
rank_2_counts = Counter()
rank_3_counts = Counter()

def get_model_name(answers, index_1_based):
    try:
        idx = index_1_based - 1
        if 0 <= idx < len(answers):
            return answers[idx]['model']
        return "Unknown"
    except:
        return "Error"

# 辅助函数：根据背景色决定文字是黑色还是白色
def get_text_color(bg_color_hex):
    rgb = mcolors.to_rgb(bg_color_hex)
    # 计算相对亮度 (Luma)
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return 'black' if luminance > 0.5 else 'white'

# 1. 读取数据
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                entry = json.loads(line)
                verify_result = entry.get('verify_result', [])
                answers = entry.get('answers', [])
                if len(verify_result) >= 3 and answers:
                    rank_1_counts[get_model_name(answers, verify_result[0])] += 1
                    rank_2_counts[get_model_name(answers, verify_result[1])] += 1
                    rank_3_counts[get_model_name(answers, verify_result[2])] += 1
            except json.JSONDecodeError:
                continue

    # 2. 数据合并逻辑
    def group_small_slices(counts, threshold):
        total = sum(counts.values())
        new_counts = {}
        others_count = 0
        for model, count in counts.items():
            if total > 0 and (count / total) * 100 < threshold:
                others_count += count
            else:
                new_counts[model] = count
        if others_count > 0:
            new_counts[OTHERS_LABEL] = others_count
        return new_counts

    r1 = group_small_slices(rank_1_counts, THRESHOLD_PERCENT)
    r2 = group_small_slices(rank_2_counts, THRESHOLD_PERCENT)
    r3 = group_small_slices(rank_3_counts, THRESHOLD_PERCENT)

    # 3. 颜色映射逻辑
    raw_models = sorted(list(set(rank_1_counts.keys()) | set(rank_2_counts.keys()) | set(rank_3_counts.keys())))
    raw_models = [m for m in raw_models if m not in ["Unknown", "Error"]]
    
    model_to_color = {}
    for i, model in enumerate(raw_models):
        # 循环使用自定义颜色列表
        color = CUSTOM_COLORS[i % len(CUSTOM_COLORS)]
        model_to_color[model] = color
    model_to_color[OTHERS_LABEL] = OTHERS_COLOR

    # --- 绘图逻辑 ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.6)) # 稍微加宽一点画布

    # # 大标题：加黑，稍微调大
    # fig.suptitle('Model Ranking Distribution', fontsize=32, fontweight='bold', y=0.96, color='#333333')

    def draw_pie(ax, counts, title):
        if not counts:
            ax.text(0.5, 0.5, 'No Data', ha='center')
            return
        
        sorted_items = sorted([item for item in counts.items() if item[0] != OTHERS_LABEL], key=lambda x: x[1], reverse=True)
        others_item = [(OTHERS_LABEL, counts[OTHERS_LABEL])] if OTHERS_LABEL in counts else []
        
        final_labels = [k for k, v in sorted_items] + [k for k, v in others_item]
        final_sizes = [v for k, v in sorted_items] + [v for k, v in others_item]
        pie_colors = [model_to_color[l] for l in final_labels]
        
        wedges, texts, autotexts = ax.pie(
            final_sizes, 
            labels=None, 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=pie_colors,
            radius=1.18, # 稍微收一点点半径，防止过于拥挤
            pctdistance=0.75, # 文字位置
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'} # 给扇区加个白边，看起来更精致
        )

        # 优化百分比文字：根据背景色自动调整黑/白，并加粗
        for text, wedge in zip(autotexts, wedges):
            text.set_fontsize(16)
            text.set_fontweight('bold')
            # 获取当前扇区颜色
            facecolor = wedge.get_facecolor() 
            # 转换为Hex以复用前面的逻辑，或者直接计算
            text_color = 'white' if (0.299*facecolor[0] + 0.587*facecolor[1] + 0.114*facecolor[2]) < 0.6 else '#333333'
            text.set_color(text_color)

        # 优化子图标题
        ax.text(
            0.5, 0, 
            title, 
            transform=ax.transAxes,
            ha='center', va='top', 
            fontsize=24, fontweight='bold', color='#333333'
        )

    draw_pie(axes[0], r1, 'Rank 1')
    draw_pie(axes[1], r2, 'Rank 2')
    draw_pie(axes[2], r3, 'Rank 3')

    # 图例设置
    legend_labels_main = [m for m in raw_models if m != OTHERS_LABEL]
    # 确保图例里包含实际出现过的模型，或者全部列出
    # 这里简单起见，列出所有 raw_models + Others
    if OTHERS_LABEL in model_to_color:
        legend_labels = legend_labels_main + [OTHERS_LABEL]
    else:
        legend_labels = legend_labels_main

    legend_handles = [mpatches.Patch(color=model_to_color[m], label=m) for m in legend_labels]
    
    # 自动计算合适的列数
    n_cols = 3 

    leg = fig.legend(
        handles=legend_handles,
        loc="lower center", 
        bbox_to_anchor=(0.5, 0.0), 
        ncol=n_cols, 
        fontsize=24, 
        title_fontsize=16,
        frameon=False, # 去掉图例边框，更像现代UI
        columnspacing=1.5 # 增加列间距
    )

    # 调整布局
    plt.subplots_adjust(
        left=0.0,    
        right=1.0,   
        top=1.0,     # 顶部留给大标题
        bottom=0.35,  # 底部留给图例
        wspace=0.0   
    )
    
    plt.savefig("model_dist.png", dpi=800) # 提高 DPI 使图片更清晰
    print("已生成图片：model_dist.png")

except FileNotFoundError:
    print(f"错误: 找不到文件 {INPUT_FILE}")