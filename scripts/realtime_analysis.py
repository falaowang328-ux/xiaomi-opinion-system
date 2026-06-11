# -*- coding: utf-8 -*-

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except:
    pass

import pandas as pd
import jieba.analyse
from snownlp import SnowNLP
import os


# =========================
# 1. 文件路径
# =========================

input_path = r"C:\Users\FALAOWANG\Desktop\xiaomi-opinion-system\data\baidu_realtime_data.csv"
output_path = r"C:\Users\FALAOWANG\Desktop\xiaomi-opinion-system\data\response_data.csv"
summary_path = r"C:\Users\FALAOWANG\Desktop\xiaomi-opinion-system\data\problem_summary.txt"


# =========================
# 2. 判断文件是否存在
# =========================

if not os.path.exists(input_path):
    print("未找到实时采集数据：", input_path)
    raise SystemExit


# =========================
# 3. 读取数据
# =========================

df = pd.read_csv(
    input_path,
    encoding="utf-8-sig"
)

df = df.fillna("")


# =========================
# 4. 风险词库
# =========================

risk_words = [
    "事故", "碰撞", "失控", "起火", "自燃",
    "死亡", "受伤", "骨折", "召回", "投诉",
    "维权", "退订", "退定", "延期", "交付",
    "故障", "异常", "断电", "刹车", "制动",
    "智驾", "辅助驾驶", "自动驾驶", "电池",
    "续航", "缩水", "售后", "质量", "安全隐患"
]


# =========================
# 5. 情感分析函数
# =========================

def get_sentiment(text):
    try:
        score = SnowNLP(str(text)).sentiments
    except:
        score = 0.5

    if score >= 0.6:
        label = "正面"
    elif score <= 0.4:
        label = "负面"
    else:
        label = "中性"

    return score, label


# =========================
# 6. 关键词提取函数
# =========================

def extract_keywords(text):
    text = str(text)

    try:
        words = jieba.analyse.extract_tags(
            text,
            topK=5
        )
    except:
        words = []

    hit_words = []

    for w in risk_words:
        if w in text:
            hit_words.append(w)

    for w in hit_words:
        if w not in words:
            words.append(w)

    stop_words = [
        "小米", "汽车", "SU7", "YU7", "用户", "百度", "新闻"
    ]

    words = [
        w for w in words
        if w not in stop_words
    ]

    return ",".join(words[:8]), hit_words


# =========================
# 7. 主题类别判断函数
# =========================

def get_cluster_name(text):
    text = str(text)

    if any(w in text for w in ["事故", "碰撞", "失控", "起火", "自燃", "刹车", "骨折", "死亡"]):
        return "安全事故类"

    elif any(w in text for w in ["智驾", "辅助驾驶", "自动驾驶", "OTA"]):
        return "智能驾驶类"

    elif any(w in text for w in ["续航", "电池", "充电", "缩水"]):
        return "续航电池类"

    elif any(w in text for w in ["交付", "延期", "退订", "退定", "订单"]):
        return "交付服务类"

    elif any(w in text for w in ["价格", "降价", "配置", "权益"]):
        return "价格配置类"

    else:
        return "品牌传播类"


# =========================
# 8. 风险评分函数
# =========================

def calculate_risk(sentiment, hit_words, cluster_name, text):
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

    for w in high_words:
        if w in str(text):
            score += 5

    return min(score, 100)


# =========================
# 9. 风险等级函数
# =========================

def get_risk_level(score):
    if score >= 80:
        return "高风险"
    elif score >= 50:
        return "中风险"
    else:
        return "低风险"


# =========================
# 10. 自动回复生成函数
# =========================

def generate_response(row):
    keywords = row["keywords"]
    risk_level = row["risk_level"]

    if risk_level == "高风险":
        return (
            f"我们已关注到用户关于“{keywords}”的相关反馈，并已第一时间启动核查机制。"
            f"小米汽车始终将用户安全和产品质量放在首位，后续将根据调查进展及时向公众说明。"
            f"感谢用户和社会各界的监督与反馈。"
        )

    elif risk_level == "中风险":
        return (
            f"感谢大家对小米汽车的关注。关于“{keywords}”相关反馈，我们会认真记录并反馈给相关团队。"
            f"如用户在使用过程中遇到具体问题，建议通过小米汽车官方客服或售后渠道联系我们。"
            f"我们将持续优化产品体验和服务质量。"
        )

    else:
        return (
            "感谢大家对小米汽车的关注与支持。"
            "我们会持续倾听用户声音，不断优化产品体验、交付效率和服务质量。"
        )


# =========================
# 11. 批量分析
# =========================

sentiment_scores = []
sentiments = []
keywords_list = []
hit_words_list = []
cluster_names = []
risk_scores = []
risk_levels = []

for text in df["content_clean"]:
    score, sentiment = get_sentiment(text)
    keywords, hit_words = extract_keywords(text)
    cluster_name = get_cluster_name(text)
    risk_score = calculate_risk(
        sentiment,
        hit_words,
        cluster_name,
        text
    )
    risk_level = get_risk_level(risk_score)

    sentiment_scores.append(score)
    sentiments.append(sentiment)
    keywords_list.append(keywords)
    hit_words_list.append(",".join(hit_words))
    cluster_names.append(cluster_name)
    risk_scores.append(risk_score)
    risk_levels.append(risk_level)

df["sentiment_score"] = sentiment_scores
df["sentiment"] = sentiments
df["keywords"] = keywords_list
df["hit_risk_words"] = hit_words_list
df["cluster_name"] = cluster_names
df["risk_score"] = risk_scores
df["risk_level"] = risk_levels
df["generated_response"] = df.apply(
    generate_response,
    axis=1
)


# =========================
# 12. 保存分析结果
# =========================

df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("实时分析结果已保存：")
print(output_path)
print("数据量：", len(df))


# =========================
# 13. 生成企业问题总结
# =========================

total = len(df)
negative_count = int((df["sentiment"] == "负面").sum())
high_risk_count = int((df["risk_level"] == "高风险").sum())
medium_risk_count = int((df["risk_level"] == "中风险").sum())

cluster_count = df["cluster_name"].value_counts().to_dict()

if len(df) > 0:
    top_problem = df["cluster_name"].value_counts().idxmax()
else:
    top_problem = "暂无明显集中问题"

summary = (
    f"本轮实时监测共采集小米汽车相关舆情 {total} 条，"
    f"其中负面舆情 {negative_count} 条，高风险舆情 {high_risk_count} 条，"
    f"中风险舆情 {medium_risk_count} 条。"
    f"从问题类型看，当前舆情主要集中在“{top_problem}”。"
    f"各类问题分布为：{cluster_count}。"
    f"建议小米汽车优先关注高风险舆情，尤其是涉及安全事故、刹车失控、投诉维权、交付延期等内容。"
    f"对于高风险问题，应及时发布官方说明；对于中风险问题，应通过客服和售后渠道进行跟进处理。"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:
    f.write(summary)

print("问题总结已保存：")
print(summary_path)
print(summary)