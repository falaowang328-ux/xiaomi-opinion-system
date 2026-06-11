from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os

from snownlp import SnowNLP
import jieba.analyse


# =========================
# 1. 创建 FastAPI 应用
# =========================

app = FastAPI(
    title="小米汽车舆情危机预警系统",
    description="品牌舆情危机 AI 预警 + 自动应对话术生成系统",
    version="1.0.0"
)


# =========================
# 2. 跨域设置
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# 3. 数据路径
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

RESPONSE_DATA_PATH = os.path.join(PROJECT_DIR, "data", "response_data.csv")
TOP_KEYWORDS_PATH = os.path.join(PROJECT_DIR, "data", "top_keywords.csv")


# =========================
# 4. 请求模型
# =========================

class ResponseRequest(BaseModel):
    content: str = ""
    risk_level: str = "低风险"
    keywords: str = ""
    sentiment: str = ""
    cluster_name: str = ""


# =========================
# 5. 数据读取函数
# =========================

def read_response_data():
    """
    读取最终结果数据 response_data.csv
    """
    if not os.path.exists(RESPONSE_DATA_PATH):
        return pd.DataFrame()

    df = pd.read_csv(RESPONSE_DATA_PATH, encoding="utf-8-sig")
    df = df.fillna("")

    if "risk_score" in df.columns:
        df["risk_score"] = pd.to_numeric(
            df["risk_score"],
            errors="coerce"
        ).fillna(0)

    return df


def read_top_keywords():
    """
    读取高频关键词 top_keywords.csv
    """
    if not os.path.exists(TOP_KEYWORDS_PATH):
        return pd.DataFrame()

    df = pd.read_csv(TOP_KEYWORDS_PATH, encoding="utf-8-sig")
    df = df.fillna("")

    return df


# =========================
# 6. 首页接口
# =========================

@app.get("/")
def root():
    return {
        "message": "小米汽车品牌舆情危机 AI 预警系统后端启动成功",
        "data_file": RESPONSE_DATA_PATH,
        "docs": "http://127.0.0.1:8000/docs"
    }


# =========================
# 7. 获取全部舆情数据
# =========================

@app.get("/api/opinions")
def get_opinions():
    df = read_response_data()

    if df.empty:
        return {
            "code": 404,
            "message": "未找到 response_data.csv 或数据为空",
            "path": RESPONSE_DATA_PATH,
            "data": []
        }

    return {
        "code": 200,
        "message": "获取舆情数据成功",
        "total": len(df),
        "data": df.to_dict(orient="records")
    }


# =========================
# 8. 获取统计数据
# =========================

@app.get("/api/statistics")
def get_statistics():
    df = read_response_data()

    if df.empty:
        return {
            "code": 404,
            "message": "未找到数据",
            "data": {}
        }

    sentiment_count = (
        df["sentiment"].value_counts().to_dict()
        if "sentiment" in df.columns else {}
    )

    risk_count = (
        df["risk_level"].value_counts().to_dict()
        if "risk_level" in df.columns else {}
    )

    platform_count = (
        df["platform"].value_counts().to_dict()
        if "platform" in df.columns else {}
    )

    cluster_count = (
        df["cluster_name"].value_counts().to_dict()
        if "cluster_name" in df.columns else {}
    )

    high_risk_count = (
        int((df["risk_level"] == "高风险").sum())
        if "risk_level" in df.columns else 0
    )

    medium_risk_count = (
        int((df["risk_level"] == "中风险").sum())
        if "risk_level" in df.columns else 0
    )

    low_risk_count = (
        int((df["risk_level"] == "低风险").sum())
        if "risk_level" in df.columns else 0
    )

    negative_count = (
        int((df["sentiment"] == "负面").sum())
        if "sentiment" in df.columns else 0
    )

    positive_count = (
        int((df["sentiment"] == "正面").sum())
        if "sentiment" in df.columns else 0
    )

    neutral_count = (
        int((df["sentiment"] == "中性").sum())
        if "sentiment" in df.columns else 0
    )

    return {
        "code": 200,
        "message": "获取统计数据成功",
        "data": {
            "total": len(df),
            "high_risk_count": high_risk_count,
            "medium_risk_count": medium_risk_count,
            "low_risk_count": low_risk_count,
            "negative_count": negative_count,
            "positive_count": positive_count,
            "neutral_count": neutral_count,
            "sentiment_count": sentiment_count,
            "risk_count": risk_count,
            "platform_count": platform_count,
            "cluster_count": cluster_count
        }
    }


# =========================
# 9. 获取中高风险预警数据
# =========================

@app.get("/api/warnings")
def get_warnings():
    df = read_response_data()

    if df.empty:
        return {
            "code": 404,
            "message": "未找到数据",
            "data": []
        }

    if "risk_level" not in df.columns:
        return {
            "code": 400,
            "message": "数据中缺少 risk_level 字段",
            "data": []
        }

    warning_df = df[
        df["risk_level"].isin(["中风险", "高风险"])
    ].copy()

    if "risk_score" in warning_df.columns:
        warning_df = warning_df.sort_values(
            by="risk_score",
            ascending=False
        )

    return {
        "code": 200,
        "message": "获取预警数据成功",
        "total": len(warning_df),
        "data": warning_df.to_dict(orient="records")
    }


# =========================
# 10. 获取高频关键词
# =========================

@app.get("/api/top-keywords")
def get_top_keywords():
    df = read_top_keywords()

    if df.empty:
        return {
            "code": 404,
            "message": "未找到 top_keywords.csv 或数据为空",
            "path": TOP_KEYWORDS_PATH,
            "data": []
        }

    return {
        "code": 200,
        "message": "获取高频关键词成功",
        "total": len(df),
        "data": df.to_dict(orient="records")
    }


# =========================
# 11. 获取风险 Top10
# =========================

@app.get("/api/top-risk")
def get_top_risk():
    df = read_response_data()

    if df.empty:
        return {
            "code": 404,
            "message": "未找到数据",
            "data": []
        }

    if "risk_score" not in df.columns:
        return {
            "code": 400,
            "message": "数据中缺少 risk_score 字段",
            "data": []
        }

    top_df = df.sort_values(
        by="risk_score",
        ascending=False
    ).head(10)

    return {
        "code": 200,
        "message": "获取风险 Top10 成功",
        "total": len(top_df),
        "data": top_df.to_dict(orient="records")
    }


# =========================
# 12. 兼容旧版本：手动生成回复
# =========================

@app.post("/api/generate-response")
def generate_response(item: ResponseRequest):
    risk_level = item.risk_level
    keywords = item.keywords if item.keywords else "相关问题"

    if risk_level == "高风险":
        generated_response = (
            f"我们已关注到用户关于“{keywords}”的相关反馈，并已第一时间启动核查机制。"
            f"小米汽车始终将用户安全和产品质量放在首位，后续将根据调查进展及时向公众说明。"
            f"感谢用户和社会各界的监督与反馈。"
        )

    elif risk_level == "中风险":
        generated_response = (
            f"感谢大家对小米汽车的关注。关于“{keywords}”相关反馈，我们会认真记录并反馈给相关团队。"
            f"如用户在使用过程中遇到具体问题，建议通过官方客服或售后渠道联系我们，"
            f"我们将持续优化产品体验和服务质量。"
        )

    else:
        generated_response = (
            f"感谢大家对小米汽车的关注与支持。"
            f"我们会持续倾听用户声音，不断优化产品体验、交付效率和服务质量。"
        )

    return {
        "code": 200,
        "message": "话术生成成功",
        "data": {
            "content": item.content,
            "sentiment": item.sentiment,
            "risk_level": risk_level,
            "cluster_name": item.cluster_name,
            "keywords": keywords,
            "generated_response": generated_response
        }
    }


# =========================
# 13. 新版：自动识别 + 自动预警 + 自动回复
# =========================

@app.post("/api/auto-generate")
def auto_generate(item: ResponseRequest):
    """
    输入一条新的舆情文本，系统自动完成：
    1. 情感分析
    2. 关键词提取
    3. 风险词识别
    4. 风险评分
    5. 风险等级判断
    6. 自动生成回应话术
    """

    content = item.content.strip()

    if not content:
        return {
            "code": 400,
            "message": "请输入舆情内容",
            "data": {}
        }

    # =====================
    # 1. 情感分析
    # =====================

    try:
        sentiment_score = SnowNLP(content).sentiments
    except:
        sentiment_score = 0.5

    if sentiment_score >= 0.6:
        sentiment = "正面"
    elif sentiment_score <= 0.4:
        sentiment = "负面"
    else:
        sentiment = "中性"

    # =====================
    # 2. 自动关键词提取
    # =====================

    try:
        keywords_list = jieba.analyse.extract_tags(
            content,
            topK=5,
            withWeight=False
        )
    except:
        keywords_list = []

    # 风险词库
    risk_words = [
        "事故", "碰撞", "失控", "起火", "自燃",
        "死亡", "受伤", "骨折", "召回", "投诉",
        "维权", "退订", "退定", "延期", "交付",
        "故障", "异常", "断电", "刹车", "制动",
        "智驾", "辅助驾驶", "自动驾驶", "电池",
        "续航", "缩水", "售后", "质量", "安全隐患"
    ]

    # 命中的风险词
    hit_words = []

    for word in risk_words:
        if word in content:
            hit_words.append(word)

    # 把风险词也合并进关键词，防止 jieba 没提取出来
    for word in hit_words:
        if word not in keywords_list:
            keywords_list.append(word)

    keywords_list = keywords_list[:8]
    keywords = ",".join(keywords_list)

    # =====================
    # 3. 自动主题判断
    # =====================

    if any(w in content for w in ["事故", "碰撞", "失控", "起火", "自燃", "刹车", "骨折", "死亡"]):
        cluster_name = "安全事故类"
    elif any(w in content for w in ["智驾", "辅助驾驶", "自动驾驶", "OTA"]):
        cluster_name = "智能驾驶类"
    elif any(w in content for w in ["续航", "电池", "充电", "缩水"]):
        cluster_name = "续航电池类"
    elif any(w in content for w in ["交付", "延期", "退订", "退定", "订单"]):
        cluster_name = "交付服务类"
    elif any(w in content for w in ["价格", "降价", "配置", "权益"]):
        cluster_name = "价格配置类"
    else:
        cluster_name = "品牌传播类"

    # =====================
    # 4. 风险评分
    # =====================

    risk_score = 0

    # 情感分
    if sentiment == "负面":
        risk_score += 40
    elif sentiment == "中性":
        risk_score += 20
    else:
        risk_score += 5

    # 风险词分
    risk_score += len(hit_words) * 8

    # 主题分
    if cluster_name == "安全事故类":
        risk_score += 25
    elif cluster_name == "智能驾驶类":
        risk_score += 18
    elif cluster_name == "续航电池类":
        risk_score += 15
    elif cluster_name == "交付服务类":
        risk_score += 12
    elif cluster_name == "价格配置类":
        risk_score += 8
    else:
        risk_score += 3

    # 高危词额外加分
    high_risk_words = [
        "事故", "失控", "起火", "自燃", "死亡",
        "受伤", "骨折", "刹车", "召回", "安全隐患"
    ]

    for word in high_risk_words:
        if word in content:
            risk_score += 5

    risk_score = min(risk_score, 100)

    # =====================
    # 5. 风险等级
    # =====================

    if risk_score >= 80:
        risk_level = "高风险"
    elif risk_score >= 50:
        risk_level = "中风险"
    else:
        risk_level = "低风险"

    # =====================
    # 6. 自动回复
    # =====================

    if risk_level == "高风险":
        generated_response = (
            f"我们已关注到用户关于“{keywords}”的相关反馈，并已第一时间启动核查机制。"
            f"小米汽车始终将用户安全和产品质量放在首位，后续将根据调查进展及时向公众说明。"
            f"感谢用户和社会各界的监督与反馈。"
        )

    elif risk_level == "中风险":
        generated_response = (
            f"感谢大家对小米汽车的关注。关于“{keywords}”相关反馈，我们会认真记录并反馈给相关团队。"
            f"如用户在使用过程中遇到具体问题，建议通过小米汽车官方客服或售后渠道联系我们。"
            f"我们将持续优化产品体验和服务质量。"
        )

    else:
        generated_response = (
            f"感谢大家对小米汽车的关注与支持。"
            f"我们会持续倾听用户声音，不断优化产品体验、交付效率和服务质量，"
            f"为用户带来更好的出行体验。"
        )

    return {
        "code": 200,
        "message": "自动分析并生成话术成功",
        "data": {
            "content": content,
            "sentiment": sentiment,
            "sentiment_score": round(float(sentiment_score), 4),
            "keywords": keywords,
            "hit_risk_words": ",".join(hit_words),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "cluster_name": cluster_name,
            "generated_response": generated_response
        }
    }
@app.post("/api/add-opinion")
def add_opinion(item: ResponseRequest):
    """
    新增一条舆情：
    1. 自动分析情感
    2. 自动提取关键词
    3. 自动计算风险等级
    4. 自动生成回复
    5. 保存到 response_data.csv
    """

    analyze_result = auto_generate(item)

    if analyze_result["code"] != 200:
        return analyze_result

    data = analyze_result["data"]

    new_row = {
        "platform": "手动新增",
        "keyword": data["keywords"],
        "title": item.content[:30],
        "content": item.content,
        "content_clean": item.content,
        "publish_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "like_count": 0,
        "comment_count": 0,
        "repost_count": 0,
        "url": "",
        "sentiment": data["sentiment"],
        "sentiment_score": data["sentiment_score"],
        "keywords": data["keywords"],
        "cluster_name": data["cluster_name"],
        "risk_score": data["risk_score"],
        "risk_level": data["risk_level"],
        "generated_response": data["generated_response"]
    }

    old_df = read_response_data()

    new_df = pd.DataFrame([new_row])

    if old_df.empty:
        final_df = new_df
    else:
        final_df = pd.concat(
            [old_df, new_df],
            ignore_index=True
        )

    final_df.to_csv(
        RESPONSE_DATA_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    return {
        "code": 200,
        "message": "舆情已新增并完成自动分析",
        "data": new_row
    }
# =========================
# 15. 企业问题总结接口
# =========================

@app.get("/api/problem-summary")
def problem_summary():
    """
    自动统计当前舆情数据，并生成面向企业的问题总结。
    """

    df = read_response_data()

    if df.empty:
        return {
            "code": 404,
            "message": "暂无数据",
            "data": {}
        }

    total = len(df)

    negative_count = (
        int((df["sentiment"] == "负面").sum())
        if "sentiment" in df.columns else 0
    )

    high_risk_count = (
        int((df["risk_level"] == "高风险").sum())
        if "risk_level" in df.columns else 0
    )

    medium_risk_count = (
        int((df["risk_level"] == "中风险").sum())
        if "risk_level" in df.columns else 0
    )

    cluster_count = (
        df["cluster_name"].value_counts().to_dict()
        if "cluster_name" in df.columns else {}
    )

    if cluster_count:
        top_problem = max(cluster_count, key=cluster_count.get)
    else:
        top_problem = "暂无明显集中问题"

    summary = (
        f"本轮监测共发现小米汽车相关舆情 {total} 条，"
        f"其中负面舆情 {negative_count} 条，高风险舆情 {high_risk_count} 条，"
        f"中风险舆情 {medium_risk_count} 条。"
        f"当前舆情主要集中在“{top_problem}”。"
        f"建议企业优先关注高风险问题，尤其是涉及安全事故、刹车失控、投诉维权、"
        f"交付延期等内容，并针对集中问题及时发布说明或优化服务流程。"
    )

    return {
        "code": 200,
        "message": "问题总结生成成功",
        "data": {
            "total": total,
            "negative_count": negative_count,
            "high_risk_count": high_risk_count,
            "medium_risk_count": medium_risk_count,
            "cluster_count": cluster_count,
            "top_problem": top_problem,
            "summary": summary
        }
    }
