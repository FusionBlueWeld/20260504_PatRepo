import json
import numpy as np
import ollama
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# プロンプトファイルの格納先
PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_prompt(filename: str, **kwargs) -> str:
    """プロンプトファイルを読み込み、プレースホルダーを埋めて返す"""
    prompt_path = PROMPTS_DIR / filename
    template = prompt_path.read_text(encoding="utf-8")
    return template.format_map(kwargs)

def generate_cluster_name(summaries: list, prompt_file: str) -> str:
    """
    LLMを用いて、クラスタに含まれる要約文リストからクラスタ名を生成する。
    prompt_file: 使用するプロンプトファイル名 (cluster_name_tech.txt or cluster_name_problem.txt)
    """
    joined_summaries = "\n".join(f"・ {s}" for s in summaries)
    prompt = load_prompt(prompt_file, joined_summaries=joined_summaries)

    try:
        response = ollama.generate(
            model='gemma4:e4b',
            prompt=prompt,
            stream=False,
            format='json',
            options={
                "num_ctx": 131072,
                "num_predict": 1024,
                "temperature": 0.0,
                "top_p": 0.9,
            }
        )
        result = json.loads(response['response'])
        name = result.get("クラスタ名", "")
        if not name:
            return "名称未設定"
        return name
    except json.JSONDecodeError:
        return "名称生成失敗 (JSONパースエラー)"
    except Exception as e:
        print(f"        [Error] LLM呼び出しエラー: {e}")
        return "名称生成失敗 (LLMエラー)"


def cluster_and_save(data_list: list, vector_key: str, summary_key: str, summary_out_key: str, output_path: Path, min_cluster_size: int, prefix: str, prompt_file: str):
    """
    ベクトルデータを用いてクラスタリングを行い、結果をJSON形式で保存する。
    指定した件数(min_cluster_size)に満たないクラスタは出力から除外する。
    抽出したクラスタごとにLLMを呼び出し、クラスタ名を生成する。
    prefix: クラスタIDの先頭に付与する文字列 (例: "p", "t")
    prompt_file: クラスタ名生成に使用するプロンプトファイル名
    """
    valid_data = [d for d in data_list if d.get(vector_key) and len(d[vector_key]) > 0]

    n_samples = len(valid_data)
    if n_samples == 0:
        print(f"    [Warn] {vector_key} の有効なデータがありません。")
        return

    vectors = np.array([d[vector_key] for d in valid_data])
    vectors_normalized = normalize(vectors, norm='l2')

    n_clusters = max(2, int(n_samples ** 0.5))
    n_clusters = min(n_clusters, n_samples)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors_normalized)

    clusters_info = {}
    for idx, label in enumerate(labels):
        if label not in clusters_info:
            clusters_info[label] = {
                "count": 0,
                "vectors": [],
                "documents": [],   # {"pdf_name": ..., "カテゴリ": ...} をまとめて管理
                "summaries": []
            }
        clusters_info[label]["count"] += 1
        clusters_info[label]["vectors"].append(valid_data[idx][vector_key])
        clusters_info[label]["documents"].append({
            "pdf_name": valid_data[idx]["pdf_name"],
            "カテゴリ":  valid_data[idx]["カテゴリ"]
        })
        clusters_info[label]["summaries"].append(valid_data[idx][summary_key])

    filtered_clusters = [c for c in clusters_info.values() if c["count"] >= min_cluster_size]
    sorted_clusters = sorted(filtered_clusters, key=lambda x: x["count"], reverse=True)

    result_json = []
    print(f"    - {output_path.name} のクラスタ名生成を開始します (対象: {len(sorted_clusters)} クラスタ)")

    for i, cluster in enumerate(sorted_clusters):
        cluster_id = f"{prefix}{i+1:03}"
        centroid = np.mean(cluster["vectors"], axis=0).tolist()

        print(f"      [{i+1}/{len(sorted_clusters)}] クラスタ {cluster_id} の名称を生成中... (構成件数: {cluster['count']}件)")
        cluster_name = generate_cluster_name(cluster["summaries"], prompt_file)

        # {"ファイル名": "カテゴリ"} の辞書リストとして出力
        documents_out = [
            {doc["pdf_name"]: doc["カテゴリ"]}
            for doc in cluster["documents"]
        ]

        result_json.append({
            "クラスタid": cluster_id,
            "クラスタ名": cluster_name,
            "クラスタに含まれる件数": cluster["count"],
            "クラスタの重心ベクトル": centroid,
            "個別文献ファイル名": documents_out,
            summary_out_key: cluster["summaries"]
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=4)

    print(f"    [Done] {output_path.name} を出力しました (出力対象: {len(result_json)} クラスタ / 除外: {len(clusters_info) - len(result_json)} クラスタ)\n")


def run_clustering(output_dir: Path, min_cluster_size: int = 5):
    """outputディレクトリ内の解析済みJSONを収集し、クラスタリングを実行する"""
    print(f"\n>>> クラスタリング処理を開始します... (出力対象の最小件数: {min_cluster_size}件)")

    visualized_dir = output_dir / "visualized" / "clustering"

    # フォルダが既に存在し、かつ中にファイル(JSON等)がある場合はスキップ
    if visualized_dir.exists() and any(visualized_dir.iterdir()):
        print(f"    [Skip] クラスタリング結果フォルダ ({visualized_dir}) が既に存在するため、処理をスキップします。\n")
        return

    visualized_dir.mkdir(parents=True, exist_ok=True)

    extracted_data = []

    patents_dir = output_dir / "patents"
    if not patents_dir.exists():
        print(f"    [Warn] 解析済みデータフォルダ ({patents_dir}) が見つかりません。")
        return

    for json_path in patents_dir.rglob("analysis.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                extracted_data.append({
                    "pdf_name": json_path.parent.name,
                    "カテゴリ":  data.get("カテゴリ", "未分類"),
                    "技術ベクトル": data.get("技術ベクトル", []),
                    "技術要約":   data.get("技術要約", ""),
                    "課題ベクトル": data.get("課題ベクトル", []),
                    "課題要約":   data.get("課題要約", "")
                })
        except Exception as e:
            print(f"    [Error] JSON読み込み失敗 ({json_path}): {e}")

    if not extracted_data:
        print("    [Warn] クラスタリング対象のデータがありません。")
        return

    # 1. 発明技術のクラスタリング
    cluster_and_save(
        data_list=extracted_data,
        vector_key="技術ベクトル",
        summary_key="技術要約",
        summary_out_key="個別文献の発明の要約文",
        output_path=visualized_dir / "clusters_tech.json",
        min_cluster_size=min_cluster_size,
        prefix="t",
        prompt_file="cluster_name_tech.txt"
    )

    # 2. 課題のクラスタリング
    cluster_and_save(
        data_list=extracted_data,
        vector_key="課題ベクトル",
        summary_key="課題要約",
        summary_out_key="個別文献の課題の要約文",
        output_path=visualized_dir / "clusters_problem.json",
        min_cluster_size=min_cluster_size,
        prefix="p",
        prompt_file="cluster_name_problem.txt"
    )

    print("<<< クラスタリング処理が完了しました。\n")