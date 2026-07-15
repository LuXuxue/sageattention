import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

def extract_category(case_name):
    match = re.match(r'^([A-Za-z]+)', str(case_name))
    return match.group(1) if match else 'Unknown'

def optimize_tradeoff_autotune(csv_file, threshold=0.05):
    # 1. 读取与清洗数据
    df = pd.read_csv(csv_file)
    df = df[(df['Time'] > 0) & (df['TFLOPS'] > 0)] # 剔除异常/报错数据
    
    # 2. 构建矩阵与基准指标
    pivot_time = df.pivot_table(index='Case', columns='config', values='Time', aggfunc='min')
    pivot_time = pivot_time.fillna(99999.0) # 缺失值视为极慢
    
    min_times = pivot_time.min(axis=1)
    
    # 【核心逻辑 1】：权重 = 历史最优延迟。延迟越高的 Case，对总体速度影响越大，权重越高
    weights = min_times 
    total_weight = weights.sum()
    
    # 构建 10% 阈值覆盖矩阵 (True 代表该 Config 能在 10% 性能损失内完成该 Case)
    regret_matrix = (pivot_time.subtract(min_times, axis=0)).divide(min_times, axis=0)
    valid_coverage = regret_matrix <= threshold
    
    # 识别“绝对无法在 10% 内覆盖”的孤岛 Case
    coverable_mask = valid_coverage.any(axis=1)
    coverable_cases = valid_coverage.index[coverable_mask].tolist()
    uncoverable_cases = valid_coverage.index[~coverable_mask].tolist()
    
    print(f"[*] 识别到 {len(coverable_cases)} 个可覆盖 Case，{len(uncoverable_cases)} 个长尾孤岛 Case。\n")
    
    # ==========================================
    # 【核心逻辑 2】：加权贪心集合覆盖 (按延迟影响力排序)
    # ==========================================
    selected_configs = []
    config_details = [] # 记录每个 Config 的覆盖详情
    cumulative_weight = 0.0
    
    uncovered_mask = pd.Series(True, index=coverable_cases)
    all_configs = pivot_time.columns.tolist()
    
    print("--- 按【延迟影响力】贪心挑选核心 Config ---")
    while uncovered_mask.any():
        best_config = None
        max_score = -1.0
        best_newly_covered = []
        
        for config in all_configs:
            if config in selected_configs:
                continue
            
            # 计算该 Config 能【新覆盖】的 Case 的权重总和
            newly_covered_mask = valid_coverage.loc[coverable_cases, config] & uncovered_mask
            score = weights[newly_covered_mask].sum()
            
            if score > max_score:
                max_score = score
                best_config = config
                best_newly_covered = weights[newly_covered_mask].index.tolist()
                
        if best_config is None or max_score <= 0:
            break # 理论上不会发生，除非数据异常
            
        selected_configs.append(best_config)
        cumulative_weight += max_score
        
        # 记录详情
        config_details.append({
            'config': best_config,
            'new_cases': best_newly_covered,
            'cumulative_pct': cumulative_weight / total_weight
        })
        
        # 更新未覆盖集合
        uncovered_mask.loc[best_newly_covered] = False

    # ==========================================
    # 【核心逻辑 3】：处理无法在 10% 内覆盖的长尾 Case (兜底补丁)
    # ==========================================
    if uncoverable_cases:
        print("--- 为长尾孤岛 Case 挑选兜底补丁 ---")
        # 找一个能让这些孤岛 Case 平均遗憾最小的 Config
        isolated_regret = regret_matrix.loc[uncoverable_cases]
        best_patch_config = isolated_regret.mean(axis=0).idxmin()
        
        selected_configs.append(best_patch_config)
        config_details.append({
            'config': best_patch_config,
            'new_cases': uncoverable_cases,
            'cumulative_pct': 1.0, # 视为 100% 兜底
            'is_patch': True
        })

    # ==========================================
    # 结果输出与 Trade-off 截断建议
    # ==========================================
    print("\n" + "="*90)
    print(f"按【延迟影响力】排序的 Config 完备列表 (阈值: {threshold*100:.0f}%)")
    print("="*90)
    
    for idx, detail in enumerate(config_details):
        c = detail['config']
        cases = detail['new_cases']
        cum_pct = detail['cumulative_pct']
        is_patch = detail.get('is_patch', False)
        
        prefix = f"[{idx+1:02d}] [兜底补丁]" if is_patch else f"[{idx+1:02d}]"
        print(f"\n{prefix} Config JSON: {c}")
        
        if is_patch:
            print(f"     |-> 核心贡献: 尽力覆盖 {len(cases)} 个【无法在{threshold*100:.0f}%内完成】的长尾 Case")
            for case in cases:
                abs_min = min_times[case]
                final_min = pivot_time.loc[case, c]
                regret = (final_min - abs_min) / abs_min * 100
                print(f"         - {case} (基准: {abs_min:.4f}ms, 当前遗憾: {regret:.1f}%)")
        else:
            print(f"     |-> 核心贡献: 新覆盖 {len(cases)} 个 Case (累计覆盖总延迟权重的 {cum_pct*100:.1f}%)")
            # 按延迟从高到低排序展示守护的 Case
            sorted_cases = sorted(cases, key=lambda x: min_times[x], reverse=True)
            for case in sorted_cases:
                print(f"         - {case} (基准延迟: {min_times[case]:.4f}ms)")

    # ==========================================
    # Trade-off 截断建议面板
    # ==========================================
    print("\n" + "="*90)
    print("Trade-off 截断建议 (自由选取列表长度，平衡 Autotune 耗时与最终性能)")
    print("="*90)
    
    milestones = [0.80, 0.90, 0.95, 0.99]
    for m in milestones:
        for i, detail in enumerate(config_details):
            if detail.get('is_patch', False):
                continue
            if detail['cumulative_pct'] >= m:
                print(f"- 选取前 [{i+1:2d}] 条 Config: 可覆盖 {detail['cumulative_pct']*100:.1f}% 的总体延迟权重 (保障核心高延迟业务)")
                break
                
    print("\n" + "="*90)
    print("纯净 Config 列表 (按重要性排序，可直接复制入代码截断使用):")
    print("="*90)
    for c in selected_configs:
        print(c)

if __name__ == "__main__":
    # threshold: 可接受的性能下降阈值，0.10 代表允许比历史最快慢 10%
    optimize_tradeoff_autotune('benchmark_attn-tune.csv', threshold=0.05)