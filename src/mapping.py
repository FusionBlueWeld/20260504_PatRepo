import json
import warnings
import logging
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import ollama
from pathlib import Path
from collections import defaultdict, Counter
from numpy.linalg import norm

# 警告およびフォント関連のログ出力を完全にミュート
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

plt.rcParams['font.family'] = ['Meiryo', 'Yu Gothic', 'MS Gothic', 'Hiragino Sans', 'sans-serif']

def get_embedding(text: str) -> list:
    try:
        response = ollama.embeddings(model='qwen3-embedding:8b', prompt=text)
        return response.get('embedding', [])
    except Exception as e:
        print(f"        [Error] Embedding取得エラー: {e}")
        return []

def calculate_similarity_and_sort(clusters: list, origin_vec: list) -> list:
    origin_arr = np.array(origin_vec)
    origin_norm = norm(origin_arr)
    for c in clusters:
        c_vec = np.array(c.get("クラスタの重心ベクトル", []))
        c_norm = norm(c_vec)
        if origin_norm == 0 or c_norm == 0:
            c["_sim"] = 0
        else:
            c["_sim"] = np.dot(origin_arr, c_vec) / (origin_norm * c_norm)
    return sorted(clusters, key=lambda x: x["_sim"], reverse=True)

def create_threat_map(output_dir: Path, mapping_dir: Path):
    print("    - 脅威マップの作成を開始します...")
    patents_dir = output_dir / "patents"
    data = []
    if patents_dir.exists():
        for json_path in patents_dir.rglob("analysis.json"):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    j = json.load(f)
                    sim = j.get("類似度", 1)
                    concept = j.get("概念高さ", 1)
                    category = j.get("カテゴリ", "未分類")
                    data.append((sim, concept, category, json_path.parent.name))
            except Exception:
                continue

    if not data: return

    min_val, max_val = 1, 5
    size = max_val - min_val + 1
    cell_data = defaultdict(Counter)
    cell_docs = defaultdict(lambda: defaultdict(list))

    for sim, concept, category, pdf_name in data:
        s = max(min_val, min(max_val, sim))
        c = max(min_val, min(max_val, concept))
        cell_data[(c, s)][category] += 1
        cell_docs[(c, s)][category].append(pdf_name)

    heatmap_data = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            c_val, s_val = i + min_val, j + min_val
            heatmap_data[i, j] = sum(cell_data.get((c_val, s_val), Counter()).values())

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor('white')

    vmax = heatmap_data.max() if heatmap_data.max() > 0 else 1
    ax.imshow(heatmap_data, cmap='jet', alpha=0.2, origin='lower', 
              extent=[0, size, 0, size], aspect='auto', vmin=0, vmax=vmax, zorder=0)

    for i in range(size):
        for j in range(size):
            c_val, s_val = i + min_val, j + min_val
            counter = cell_data.get((c_val, s_val), Counter())
            total_count = sum(counter.values())

            if total_count > 0:
                ax.text(j + 0.5, i + 0.53, f"{total_count}件", ha='center', va='bottom', color="black", fontsize=14, fontweight="bold", zorder=3)
                details = [f"{cat}: {cnt}件" for cat, cnt in sorted(counter.items())]
                ax.text(j + 0.5, i + 0.47, "\n".join(details), ha='center', va='top', color="black", fontsize=9, linespacing=1.3, zorder=3)
            else:
                ax.text(j + 0.5, i + 0.5, "0件", ha='center', va='center', color="gray", fontsize=14, zorder=3)

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)

    ticks = np.arange(0.5, size, 1)
    labels = [str(v) for v in range(min_val, max_val + 1)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=12)

    ax.set_xlabel("類似度", fontsize=14, labelpad=10)
    ax.set_ylabel("概念高さ", fontsize=14, labelpad=10)
    ax.set_title("脅威マップ", fontsize=16, pad=15)
    ax.set_xticks(np.arange(0, size + 1), minor=True)
    ax.set_yticks(np.arange(0, size + 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='--', linewidth=1, zorder=2)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
        spine.set_zorder(4)

    ax.tick_params(which='both', bottom=False, left=False)
    
    plt.tight_layout()
    fig.canvas.draw()
    bbox = ax.get_position()
    
    meta_data = {
        "x_size": size, "y_size": size,
        "bbox": {"left": bbox.x0, "right": bbox.x1, "bottom": bbox.y0, "top": bbox.y1},
        "cells": {}
    }
    for (i, j), cats in cell_docs.items():
        meta_data["cells"][f"{j - min_val},{i - min_val}"] = cats
        
    with open(mapping_dir / "threat_data.json", "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)

    output_path = mapping_dir / "threat_map.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"      -> {output_path.name} を保存しました。")

def create_tech_map(output_dir: Path, mapping_dir: Path):
    print("    - 技術マップの作成を開始します...")
    cluster_dir = output_dir / "visualized" / "clustering"
    if not ((cluster_dir / "clusters_tech.json").exists() and (cluster_dir / "clusters_problem.json").exists()):
        return

    with open(cluster_dir / "clusters_tech.json", "r", encoding="utf-8") as f:
        tech_clusters = json.load(f)
    with open(cluster_dir / "clusters_problem.json", "r", encoding="utf-8") as f:
        prob_clusters = json.load(f)

    company_text = (Path(__file__).parent / "prompts" / "company_tech.txt").read_text(encoding="utf-8")
    origin_vec = get_embedding(company_text)

    sorted_tech = calculate_similarity_and_sort(tech_clusters, origin_vec)
    sorted_prob = calculate_similarity_and_sort(prob_clusters, origin_vec)

    doc_to_tech = {}
    for idx, cluster in enumerate(sorted_tech):
        for doc_dict in cluster.get("個別文献ファイル名", []):
            for doc_name, category in doc_dict.items():
                doc_to_tech[doc_name] = (idx, category)

    doc_to_prob = {}
    for idx, cluster in enumerate(sorted_prob):
        for doc_dict in cluster.get("個別文献ファイル名", []):
            for doc_name, category in doc_dict.items():
                doc_to_prob[doc_name] = idx

    cell_data = defaultdict(Counter)
    cell_docs = defaultdict(lambda: defaultdict(list))
    for doc_name, (tech_idx, category) in doc_to_tech.items():
        if doc_name in doc_to_prob:
            prob_idx = doc_to_prob[doc_name]
            cell_data[(prob_idx, tech_idx)][category] += 1
            cell_docs[(prob_idx, tech_idx)][category].append(doc_name)

    x_size, y_size = len(sorted_tech), len(sorted_prob)
    if x_size == 0 or y_size == 0: return

    heatmap_data = np.zeros((y_size, x_size))
    for i in range(y_size):
        for j in range(x_size):
            heatmap_data[i, j] = sum(cell_data.get((i, j), Counter()).values())

    fig_width = max(12, x_size * 2.8)
    fig_height = max(8, y_size * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor('white')

    vmax = heatmap_data.max() if heatmap_data.max() > 0 else 1
    ax.imshow(heatmap_data, cmap='jet', alpha=0.2, origin='upper', 
              extent=[0, x_size, y_size, 0], aspect='auto', vmin=0, vmax=vmax, zorder=0)

    for i in range(y_size):
        for j in range(x_size):
            counter = cell_data.get((i, j), Counter())
            total_count = sum(counter.values())
            if total_count > 0:
                ax.text(j + 0.5, i + 0.45, f"{total_count}件", ha='center', va='bottom', color="black", fontsize=12, fontweight="bold", zorder=3)
                details = [f"{cat}: {cnt}件" for cat, cnt in sorted(counter.items())]
                ax.text(j + 0.5, i + 0.55, "\n".join(details), ha='center', va='top', color="black", fontsize=9, linespacing=1.2, zorder=3)
            else:
                ax.text(j + 0.5, i + 0.5, "0件", ha='center', va='center', color="gray", fontsize=12, zorder=3)

    ax.set_xlim(0, x_size)
    ax.set_ylim(y_size, 0)

    x_labels = [c["クラスタ名"] for c in sorted_tech]
    y_labels = [c["クラスタ名"] for c in sorted_prob]
    
    # グラフ表示用は改行付き
    x_labels_wrap = [textwrap.fill(l, width=10) for l in x_labels]
    y_labels_wrap = [textwrap.fill(l, width=15) for l in y_labels]
    
    ax.set_xticks(np.arange(0.5, x_size, 1))
    ax.set_xticklabels(x_labels_wrap, fontsize=9, linespacing=1.3)
    ax.set_yticks(np.arange(0.5, y_size, 1))
    ax.set_yticklabels(y_labels_wrap, fontsize=10, linespacing=1.2)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel("発明の技術クラスタ (左ほど自社技術に近い)", fontsize=14, labelpad=20, fontweight="bold")
    ax.set_ylabel("技術的課題クラスタ (上ほど自社技術に近い)", fontsize=14, labelpad=20, fontweight="bold")

    ax.set_xticks(np.arange(0, x_size + 1), minor=True)
    ax.set_yticks(np.arange(0, y_size + 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='--', linewidth=1, zorder=2)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
        spine.set_zorder(4)
    ax.tick_params(which='both', bottom=False, top=False, left=False)

    plt.tight_layout()
    fig.canvas.draw()
    bbox = ax.get_position()
    
    meta_data = {
        "x_size": x_size, "y_size": y_size,
        "x_labels": x_labels,  # LLM向けに改行なしのクラスタ名を保存
        "y_labels": y_labels,
        "bbox": {"left": bbox.x0, "right": bbox.x1, "bottom": bbox.y0, "top": bbox.y1},
        "cells": {}
    }
    for (i, j), cats in cell_docs.items():
        meta_data["cells"][f"{j},{i}"] = cats

    with open(mapping_dir / "tech_data.json", "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)

    output_path = mapping_dir / "tech_map.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      -> {output_path.name} を保存しました。")

def export_markdown_for_llm(mapping_dir: Path):
    """LLMに読ませるためのマークダウンテーブルを出力する"""
    print("    - LLM用Markdownデータを出力します...")
    
    md_lines = []
    md_lines.append("# Patent Mapping Data")
    md_lines.append("This document provides structured data for the Threat Map and the Technology Map.\n")
    
    # 1. 脅威マップ
    threat_json = mapping_dir / "threat_data.json"
    if threat_json.exists():
        with open(threat_json, "r", encoding="utf-8") as f:
            threat_data = json.load(f)
            
        md_lines.append("## 1. Threat Map (脅威マップ)")
        md_lines.append("- X-axis: Similarity to our company (1 to 5)")
        md_lines.append("- Y-axis: Concept Level / Broadness (1 to 5)")
        md_lines.append("- Cell format: `Total Count (CategoryA: count, ...)`\n")
        
        headers = ["Concept Level \\ Similarity"] + [str(i) for i in range(1, 6)]
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        
        # 概念高さ(Y軸)は上から 5 -> 1 と降順に並べた方が表として直感的
        for y in reversed(range(5)):
            row = [f"Level {y + 1}"]
            for x in range(5):
                cell_key = f"{x},{y}"
                cats = threat_data["cells"].get(cell_key, {})
                if not cats:
                    row.append("0")
                else:
                    total = sum(len(docs) for docs in cats.values())
                    details = ", ".join([f"{k}:{len(v)}" for k, v in cats.items()])
                    row.append(f"{total} ({details})")
            md_lines.append("| " + " | ".join(row) + " |")
        md_lines.append("\n")

    # 2. 技術マップ
    tech_json = mapping_dir / "tech_data.json"
    if tech_json.exists():
        with open(tech_json, "r", encoding="utf-8") as f:
            tech_data = json.load(f)
            
        x_labels = tech_data.get("x_labels", [])
        y_labels = tech_data.get("y_labels", [])
        
        if x_labels and y_labels:
            md_lines.append("## 2. Tech Map (技術マップ)")
            md_lines.append("- X-axis: Technical Solutions Clusters (Sorted by relevance to our technology, left is closer)")
            md_lines.append("- Y-axis: Technical Problems Clusters (Sorted by relevance to our technology, top is closer)")
            md_lines.append("- Cell format: `Total Count (CategoryA: count, ...)`\n")
            
            headers = ["Problem \\ Tech"] + x_labels
            md_lines.append("| " + " | ".join(headers) + " |")
            md_lines.append("|" + "|".join(["---"] * len(headers)) + "|")
            
            for y, y_label in enumerate(y_labels):
                row = [y_label]
                for x, x_label in enumerate(x_labels):
                    cell_key = f"{x},{y}"
                    cats = tech_data["cells"].get(cell_key, {})
                    if not cats:
                        row.append("0")
                    else:
                        total = sum(len(docs) for docs in cats.values())
                        details = ", ".join([f"{k}:{len(v)}" for k, v in cats.items()])
                        row.append(f"{total} ({details})")
                md_lines.append("| " + " | ".join(row) + " |")
            md_lines.append("\n")

    md_path = mapping_dir / "maps_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"      -> {md_path.name} を保存しました。")

def run_mapping(output_dir: Path):
    mapping_dir = output_dir / "visualized" / "mapping"
    req_files = ["threat_map.png", "tech_map.png", "threat_data.json", "tech_data.json", "maps_summary.md"]
    if all((mapping_dir / f).exists() for f in req_files):
        print("    [Skip] マップ画像およびデータが既に存在するため、マッピング処理をスキップします。\n")
        return

    print("\n>>> マッピングと可視化処理を開始します...")
    mapping_dir.mkdir(parents=True, exist_ok=True)
    create_threat_map(output_dir, mapping_dir)
    create_tech_map(output_dir, mapping_dir)
    export_markdown_for_llm(mapping_dir)
    print("<<< マッピング処理が完了しました。\n")