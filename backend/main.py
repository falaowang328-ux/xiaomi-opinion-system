# -*- coding: utf-8 -*-

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from collections import Counter
import csv
import os


app = FastAPI(
    title="小米汽车舆情危机预警系统",
    description="品牌舆情危机 AI 预警 + 自动应对话术生成系统",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

RESPONSE_DATA_PATH = os.path.join(PROJECT_DIR, "data", "response_data.csv")
TOP_KEYWORDS_PATH = os.path.join(PROJECT_DIR, "data", "top_keywords.csv")
SUMMARY_PATH = os.path.join(PROJECT_DIR, "data", "problem_summary.txt")


class ResponseRequest(BaseModel):
    content: str = ""
    risk_level: str = "低风险"
    keywords: str = ""
    sentiment: str = ""
    cluster_name: str = ""


RISK_WORDS = [
    "事故", "碰撞", "失控", "起火", "自燃",
    "死亡", "受伤", "骨折", "召回", "投诉",
    "维权", "退订", "退定", "延期", "交付",
    "故障", "异常", "断电", "刹车", "制动",
    "智驾", "辅助驾驶", "自动驾驶", "电池",
    "续航", "缩水", "售后", "质量", "安全隐患"
]

NEGATIVE_WORDS = [
    "投诉", "失控", "事故", "碰撞", "起火", "自燃",
    "故障", "刹车", "维权", "延期", "退订", "售后差",
    "质量问题", "安全隐患", "召回", "受伤", "骨折"
]

POSITIVE_WORDS = [
    "满意", "优秀", "喜欢", "好评", "提升", "优化",
    "支持", "认可", "顺利", "值得", "稳定"
]


def read_csv_data(path):
    if not os.path.exists(path):
        return []

    rows = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            clean_row = {}

            for k, v in row.items():
                if k is None:
                    continue
                clean_row[k] = "" if v is None else str(v)

            if "risk_score" in clean_row:
                try:
                    clean_row["risk_score"] = float(clean_row["risk_score"])
                except:
                    clean_row["risk_score"] = 0

            rows.append(clean_row)

    return rows


def write_csv_data(path, rows):
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = list(rows[0].keys())

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def count_by_field(rows, field):
    values = []

    for row in rows:
        value = row.get(field, "")
        if value:
            values.append(value)

    return dict(Counter(values))


def get_float(value, default=0):
    try:
        return float(value)
    except:
        return default


def simple_sentiment(content):
    neg_count = sum(1 for w in NEGATIVE_WORDS if w in content)
    pos_count = sum(1 for w in POSITIVE_WORDS if w in content)

    if neg_count > pos_count:
        return "负面", 0.2
    elif pos_count > neg_count:
        return "正面", 0.8
    else:
        return "中性", 0.5


def extract_keywords_simple(content):
    hit_words = []

    for w in RISK_WORDS:
        if w in content and w not in hit_words:
            hit_words.append(w)

    extra_words = []

    common_words = [
        "小米SU7", "小米汽车", "SU7", "YU7", "雷军",
        "交付", "售后", "价格", "配置", "智驾", "续航", "电池"
    ]

    for w in common_words:
        if w in content and w not in hit_words and w not in extra_words:
            extra_words.append(w)

    keywords = hit_words + extra_words

    if not keywords:
        keywords = ["相关问题"]

    return ",".join(keywords[:8]), hit_words


def get_cluster_name(content):
    if any(w in content for w in ["事故", "碰撞", "失控", "起火", "自燃", "刹车", "骨折", "死亡"]):
        return "安全事故类"

    if any(w in content for w in ["智驾", "辅助驾驶", "自动驾驶", "OTA"]):
        return "智能驾驶类"

    if any(w in content for w in ["续航", "电池", "充电", "缩水"]):
        return "续航电池类"

    if any(w in content for w in ["交付", "延期", "退订", "退定", "订单"]):
        return "交付服务类"

    if any(w in content for w in ["价格", "降价", "配置", "权益"]):
        return "价格配置类"

    return "品牌传播类"


def calculate_risk_score(sentiment, hit_words, cluster_name, content):
    score = 0

    if sentiment == "负面":
        score += 40
    elif sentiment == "中性":
        score += 20
    else:
        score += 5

    score += len(hit_words) * 8

    if cluster_name == "安全事故类":
        score += 25
    elif cluster_name == "智能驾驶类":
        score += 18
    elif cluster_name == "续航电池类":
        score += 15
    elif cluster_name == "交付服务类":
        score += 12
    elif cluster_name == "价格配置类":
        score += 8
    else:
        score += 3

    high_words = [
        "事故", "失控", "起火", "自燃", "死亡",
        "受伤", "骨折", "刹车", "召回", "安全隐患"
    ]

    for word in high_words:
        if word in content:
            score += 5

    return min(score, 100)


def get_risk_level(score):
    if score >= 80:
        return "高风险"
    elif score >= 50:
        return "中风险"
    else:
        return "低风险"


def generate_response_text(keywords, risk_level):
    if risk_level == "高风险":
        return (
            f"我们已关注到用户关于“{keywords}”的相关反馈，并已第一时间启动核查机制。"
            f"小米汽车始终将用户安全和产品质量放在首位，后续将根据调查进展及时向公众说明。"
            f"感谢用户和社会各界的监督与反馈。"
        )

    if risk_level == "中风险":
        return (
            f"感谢大家对小米汽车的关注。关于“{keywords}”相关反馈，我们会认真记录并反馈给相关团队。"
            f"如用户在使用过程中遇到具体问题，建议通过小米汽车官方客服或售后渠道联系我们。"
            f"我们将持续优化产品体验和服务质量。"
        )

    return (
        "感谢大家对小米汽车的关注与支持。"
        "我们会持续倾听用户声音，不断优化产品体验、交付效率和服务质量。"
    )


def analyze_content(content):
    content = content.strip()

    sentiment, sentiment_score = simple_sentiment(content)
    keywords, hit_words = extract_keywords_simple(content)
    cluster_name = get_cluster_name(content)
    risk_score = calculate_risk_score(sentiment, hit_words, cluster_name, content)
    risk_level = get_risk_level(risk_score)
    generated_response = generate_response_text(keywords, risk_level)

    return {
        "content": content,
        "sentiment": sentiment,
        "sentiment_score": round(sentiment_score, 4),
        "keywords": keywords,
        "hit_risk_words": ",".join(hit_words),
        "cluster_name": cluster_name,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "generated_response": generated_response
    }


@app.get("/")
def root():
    return {
        "message": "小米汽车品牌舆情危机 AI 预警系统后端启动成功",
        "data_file": RESPONSE_DATA_PATH,
        "docs": "/docs"
    }


@app.get("/api/opinions")
def get_opinions():
    rows = read_csv_data(RESPONSE_DATA_PATH)

    if not rows:
        return {
            "code": 404,
            "message": "未找到 response_data.csv 或数据为空",
            "path": RESPONSE_DATA_PATH,
            "data": []
        }

    return {
        "code": 200,
        "message": "获取舆情数据成功",
        "total": len(rows),
        "data": rows
    }


@app.get("/api/statistics")
def get_statistics():
    rows = read_csv_data(RESPONSE_DATA_PATH)

    if not rows:
        return {
            "code": 404,
            "message": "未找到数据",
            "data": {}
        }

    total = len(rows)

    high_risk_count = sum(1 for r in rows if r.get("risk_level") == "高风险")
    medium_risk_count = sum(1 for r in rows if r.get("risk_level") == "中风险")
    low_risk_count = sum(1 for r in rows if r.get("risk_level") == "低风险")

    negative_count = sum(1 for r in rows if r.get("sentiment") == "负面")
    positive_count = sum(1 for r in rows if r.get("sentiment") == "正面")
    neutral_count = sum(1 for r in rows if r.get("sentiment") == "中性")

    sentiment_count = count_by_field(rows, "sentiment")
    risk_count = count_by_field(rows, "risk_level")
    platform_count = count_by_field(rows, "platform")
    cluster_count = count_by_field(rows, "cluster_name")

    return {
        "code": 200,
        "message": "获取统计数据成功",
        "data": {
            "total": total,
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


@app.get("/api/warnings")
def get_warnings():
    rows = read_csv_data(RESPONSE_DATA_PATH)

    warning_rows = [
        r for r in rows
        if r.get("risk_level") in ["高风险", "中风险"]
    ]

    warning_rows = sorted(
        warning_rows,
        key=lambda x: get_float(x.get("risk_score", 0)),
        reverse=True
    )

    return {
        "code": 200,
        "message": "获取预警数据成功",
        "total": len(warning_rows),
        "data": warning_rows
    }


@app.get("/api/top-keywords")
def get_top_keywords():
    rows = read_csv_data(TOP_KEYWORDS_PATH)

    if not rows:
        return {
            "code": 404,
            "message": "未找到 top_keywords.csv 或数据为空",
            "path": TOP_KEYWORDS_PATH,
            "data": []
        }

    return {
        "code": 200,
        "message": "获取高频关键词成功",
        "total": len(rows),
        "data": rows
    }


@app.get("/api/top-risk")
def get_top_risk():
    rows = read_csv_data(RESPONSE_DATA_PATH)

    top_rows = sorted(
        rows,
        key=lambda x: get_float(x.get("risk_score", 0)),
        reverse=True
    )[:10]

    return {
        "code": 200,
        "message": "获取风险 Top10 成功",
        "total": len(top_rows),
        "data": top_rows
    }


@app.post("/api/generate-response")
def generate_response(item: ResponseRequest):
    keywords = item.keywords if item.keywords else "相关问题"
    risk_level = item.risk_level if item.risk_level else "低风险"

    generated_response = generate_response_text(keywords, risk_level)

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


@app.post("/api/auto-generate")
def auto_generate(item: ResponseRequest):
    content = item.content.strip()

    if not content:
        return {
            "code": 400,
            "message": "请输入舆情内容",
            "data": {}
        }

    data = analyze_content(content)

    return {
        "code": 200,
        "message": "自动分析并生成话术成功",
        "data": data
    }


@app.post("/api/add-opinion")
def add_opinion(item: ResponseRequest):
    content = item.content.strip()

    if not content:
        return {
            "code": 400,
            "message": "请输入舆情内容",
            "data": {}
        }

    data = analyze_content(content)

    new_row = {
        "platform": "手动新增",
        "keyword": data["keywords"],
        "title": content[:30],
        "content": content,
        "content_clean": content,
        "publish_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "like_count": "0",
        "comment_count": "0",
        "repost_count": "0",
        "url": "",
        "sentiment": data["sentiment"],
        "sentiment_score": str(data["sentiment_score"]),
        "keywords": data["keywords"],
        "hit_risk_words": data["hit_risk_words"],
        "cluster_name": data["cluster_name"],
        "risk_score": str(data["risk_score"]),
        "risk_level": data["risk_level"],
        "generated_response": data["generated_response"]
    }

    rows = read_csv_data(RESPONSE_DATA_PATH)
    rows.append(new_row)

    write_csv_data(RESPONSE_DATA_PATH, rows)

    return {
        "code": 200,
        "message": "舆情已新增并完成自动分析",
        "data": new_row
    }


@app.get("/api/problem-summary")
def problem_summary():
    rows = read_csv_data(RESPONSE_DATA_PATH)

    if not rows:
        return {
            "code": 404,
            "message": "暂无数据",
            "data": {}
        }

    total = len(rows)

    negative_count = sum(1 for r in rows if r.get("sentiment") == "负面")
    high_risk_count = sum(1 for r in rows if r.get("risk_level") == "高风险")
    medium_risk_count = sum(1 for r in rows if r.get("risk_level") == "中风险")

    cluster_count = count_by_field(rows, "cluster_name")

    if cluster_count:
        top_problem = max(cluster_count, key=cluster_count.get)
    else:
        top_problem = "暂无明显集中问题"

    summary = (
        f"本轮实时监测共发现小米汽车相关舆情{total}条，"
        f"其中负面舆情{negative_count}条，高风险舆情{high_risk_count}条，"
        f"中风险舆情{medium_risk_count}条。"
        f"当前舆情主要集中在“{top_problem}”。"
        f"建议企业优先关注高风险问题，并针对集中问题及时发布说明或优化服务流程。"
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
