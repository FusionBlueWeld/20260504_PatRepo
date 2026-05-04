import json

class TextLengthError(Exception):
    """テキストが制限文字数を超えた場合に送出される例外"""
    pass

def validate_analysis_result(raw_result: dict, target_key: str) -> dict:
    """
    類似度や概念高さのJSON形式と値を検証・補正する汎用バリデーター
    """
    validated = {
        target_key: 0,
        "根拠": "解析結果のフォーマットが不正です。"
    }

    try:
        if "similar_claims" in raw_result and isinstance(raw_result["similar_claims"], list):
            first_claim = raw_result["similar_claims"][0]
            score_raw = first_claim.get("similarity_score", first_claim.get("score", 0))
            validated["根拠"] = first_claim.get("reasoning", "")
        else:
            score_raw = raw_result.get(target_key, raw_result.get(target_key.replace("高さ", "_score"), 0))
            validated["根拠"] = raw_result.get("根拠", raw_result.get("理由", raw_result.get("reasoning", "根拠なし")))

        if isinstance(score_raw, str):
            if "高" in score_raw or "5" in score_raw: score_raw = 5
            elif "4" in score_raw: score_raw = 4
            elif "3" in score_raw: score_raw = 3
            elif "2" in score_raw: score_raw = 2
            elif "1" in score_raw: score_raw = 1
            else: score_raw = 0
        
        try:
            score_int = int(float(score_raw))
            if score_int > 5: score_int = 5
            if score_int < 0: score_int = 0
            validated[target_key] = score_int
        except:
            validated[target_key] = 0

    except Exception as e:
        validated["根拠"] = f"バリデーションエラー: {str(e)}"

    return validated

def validate_summary(raw_result: dict, target_key: str) -> str:
    """
    課題要約や技術要約のJSONを検証し、20000字を超えている場合はエラーを出す
    """
    # LLMのキーの揺らぎを吸収
    if target_key == "課題要約":
        summary = raw_result.get("課題要約", raw_result.get("要約", raw_result.get("課題", "")))
    elif target_key == "技術要約":
        summary = raw_result.get("技術要約", raw_result.get("要約", raw_result.get("技術", raw_result.get("発明の技術", ""))))
    else:
        summary = raw_result.get(target_key, "")
    
    if not isinstance(summary, str):
        summary = str(summary)

    # 20000文字を超えているかチェック
    if len(summary) > 20000:
        raise TextLengthError(f"{target_key}が規定文字数を超えています ({len(summary)}文字)")
    
    # 全く抽出できなかった場合のフォールバック
    if not summary:
        summary = f"{target_key}の抽出に失敗しました。"

    return summary