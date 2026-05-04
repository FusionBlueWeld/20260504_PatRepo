import json
import ollama
from pathlib import Path
from src.validator import validate_analysis_result, validate_summary

# プロンプトファイルの格納先
PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_prompt(filename: str, **kwargs) -> str:
    """プロンプトファイルを読み込み、プレースホルダーを埋めて返す"""
    prompt_path = PROMPTS_DIR / filename
    template = prompt_path.read_text(encoding="utf-8")
    return template.format_map(kwargs)

def call_ollama(prompt: str) -> dict:
    """Ollamaを呼び出す共通関数"""
    try:
        response = ollama.generate(
            model='gemma4:e4b',
            prompt=prompt,
            stream=False,
            format='json',
            options={
                "num_ctx": 131072,
                "num_predict": 4096,
                "temperature": 0,
                "top_p": 0.9,
            }
        )
        return json.loads(response['response'])
    except json.JSONDecodeError:
        print(f"    [Error] JSONパース失敗。")
        return {}
    except Exception as e:
        print(f"    [Error] Ollama呼び出しエラー: {e}")
        return {}

def get_embedding(text: str) -> list:
    """テキストをOllama経由でベクトル化する"""
    if not text or "失敗しました" in text:
        return []
    try:
        response = ollama.embeddings(
            model='qwen3-embedding:8b',
            prompt=text
        )
        return response.get('embedding', [])
    except Exception as e:
        print(f"    [Error] Embedding取得エラー: {e}")
        return []

def analyze_similarity(target_text: str) -> dict:
    company_tech = load_prompt("company_tech.txt")
    prompt = load_prompt("similarity.txt", company_tech=company_tech, target_text=target_text)
    raw_json = call_ollama(prompt)
    return validate_analysis_result(raw_json, "類似度")

def analyze_concept_level(target_text: str) -> dict:
    prompt = load_prompt("concept_level.txt", target_text=target_text)
    raw_json = call_ollama(prompt)
    return validate_analysis_result(raw_json, "概念高さ")

def analyze_problem(target_text: str) -> str:
    prompt = load_prompt("problem.txt", target_text=target_text)
    raw_json = call_ollama(prompt)
    return validate_summary(raw_json, "課題要約")

def analyze_tech(target_text: str) -> str:
    prompt = load_prompt("tech.txt", target_text=target_text)
    raw_json = call_ollama(prompt)
    return validate_summary(raw_json, "技術要約")

def analyze_patent(patent_text: str) -> tuple:
    """統合分析関数（JSON用データとCLI表示用メタ情報を返す）"""
    LIMIT = 100000
    original_len = len(patent_text)
    is_clipped = original_len > LIMIT
    target_text = patent_text[:LIMIT]

    sim_result = analyze_similarity(target_text)
    concept_result = analyze_concept_level(target_text)
    problem_summary = analyze_problem(target_text)
    tech_summary = analyze_tech(target_text)

    problem_embedding = get_embedding(problem_summary)
    tech_embedding = get_embedding(tech_summary)

    json_data = {
        "類似度": sim_result["類似度"],
        "類似度根拠": sim_result["根拠"],
        "概念高さ": concept_result["概念高さ"],
        "概念高さ根拠": concept_result["根拠"],
        "課題要約": problem_summary,
        "課題ベクトル": problem_embedding,
        "技術要約": tech_summary,
        "技術ベクトル": tech_embedding
    }

    meta_data = {
        "is_clipped": is_clipped,
        "original_len": original_len
    }

    return json_data, meta_data