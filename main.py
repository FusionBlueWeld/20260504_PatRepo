import json
import time
import shutil
from pathlib import Path
from src.patent_extractor import extract_text_from_pdf
from src.patent_analyzer import analyze_patent
from src.validator import TextLengthError
from src.clustering import run_clustering
from src.mapping import run_mapping
from gui.app import run_gui  # ← 追加

def main():
    data_dir = Path("data")
    output_base_dir = Path("output")
    mapping_dir = output_base_dir / "visualized" / "mapping"

    # ファイルが全て揃っているかチェック
    req_files = ["threat_map.png", "tech_map.png", "threat_data.json", "tech_data.json"]
    if all((mapping_dir / f).exists() for f in req_files):
        print(">>> マップ画像とデータが既に存在するため、解析・描画処理をスキップしGUIを起動します。\n")
        run_gui(output_base_dir)
        return

    patents_dir = output_base_dir / "patents"

    # PDFファイル一覧を再帰的に取得
    pdf_files = list(data_dir.rglob("*.pdf"))
    if not pdf_files:
        print("PDFファイルが見つかりません。")
        return

    print(f"全 {len(pdf_files)} 件のチェックを開始します。\n")

    for index, pdf_path in enumerate(pdf_files):
        pdf_name = pdf_path.stem
        category = pdf_path.parent.name
        if category == data_dir.name:
            category = "未分類"

        target_dir = patents_dir / pdf_name
        json_path = target_dir / "analysis.json"

        if json_path.exists():
            print(f"[{index+1}/{len(pdf_files)}] スキップ: {pdf_name} (解析済み)")
            continue

        print(f"[{index+1}/{len(pdf_files)}] >>> 処理開始: {pdf_name}")
        target_dir.mkdir(parents=True, exist_ok=True)

        text_path = target_dir / f"{pdf_name}.txt"
        try:
            if text_path.exists():
                text = text_path.read_text(encoding="utf-8")
            else:
                text = extract_text_from_pdf(pdf_path)
                text_path.write_text(text, encoding="utf-8")
        except Exception as e:
            print(f"    [Error] テキスト抽出失敗: {e}")
            continue

        try:
            analysis_result, meta_data = analyze_patent(text)
            json_output = {"カテゴリ": category}
            json_output.update(analysis_result)
            
            print(f"    カテゴリ: {category}")
            print(f"    入力クリップ発生: {str(meta_data['is_clipped']).lower()},")
            print(f"    元テキスト文字数: {meta_data['original_len']}")
            print(f"    [Done] 類似度: {analysis_result['類似度']}")
            print(f"    [Done] 概念高さ: {analysis_result['概念高さ']}")
            print(f"    [Done] 課題/技術 要約")
            print(f"    [Done] Embedding ベクトル化")
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_output, f, ensure_ascii=False, indent=4)

            if index < len(pdf_files) - 1:
                print("    60秒間の待機中...\n")
                time.sleep(60)
            else:
                print()

        except TextLengthError as e:
            print(f"    [Skip] {e}")
            print(f"    [Skip] 異常な文字数のため、出力フォルダ({pdf_name})を削除してスキップします。\n")
            if target_dir.exists():
                shutil.rmtree(target_dir)
            continue
            
        except Exception as e:
            print(f"    [Error] 分析失敗: {e}\n")

    print("すべてのテキスト解析処理が完了しました。")
    
    run_clustering(output_base_dir, min_cluster_size=5)
    run_mapping(output_base_dir)
    
    print("すべての処理が完了しました。GUIを起動します。")
    run_gui(output_base_dir)

if __name__ == "__main__":
    main()