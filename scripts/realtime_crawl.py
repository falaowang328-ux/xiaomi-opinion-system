# -*- coding: utf-8 -*-

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except:
    pass

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import os
from urllib.parse import quote
from datetime import datetime


# =========================
# 1. 数据保存路径
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "baidu_realtime_data.csv")


# =========================
# 2. 文本清洗函数
# =========================

def clean_text(text):
    if text is None:
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ").replace("\t", " ")

    return text.strip()


# =========================
# 3. 百度新闻实时采集函数
# =========================

def crawl_baidu_news(keyword="小米汽车", pages=1):
    """
    从百度新闻搜索结果中实时采集小米汽车相关舆情
    keyword：搜索关键词
    pages：采集页数
    """

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Edge/120.0 Safari/537.36"
    ]

    data = []

    for page in range(pages):
        pn = page * 10

        url = (
            "https://www.baidu.com/s?"
            f"tn=news"
            f"&rtt=1"
            f"&bsst=1"
            f"&wd={quote(keyword)}"
            f"&pn={pn}"
        )

        headers = {
            "User-Agent": random.choice(user_agents),
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }

        print(f"正在采集：{keyword} 第 {page + 1} 页")

        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=15
            )

            response.encoding = "utf-8"

            soup = BeautifulSoup(response.text, "html.parser")

            results = soup.select("div.result, div.c-container")

            for item in results:
                title_tag = item.find("h3")

                if not title_tag:
                    continue

                a_tag = title_tag.find("a")

                title = clean_text(
                    title_tag.get_text(" ", strip=True)
                )

                link = a_tag.get("href", "") if a_tag else ""

                content = clean_text(
                    item.get_text(" ", strip=True)
                )

                if len(title) < 4:
                    continue

                related_words = [
                    "小米", "SU7", "YU7", "雷军", "小米汽车"
                ]

                if not any(w in content for w in related_words):
                    continue

                data.append({
                    "platform": "百度新闻",
                    "keyword": keyword,
                    "title": title,
                    "content": content,
                    "content_clean": content,
                    "publish_time": "",
                    "crawl_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "like_count": 0,
                    "comment_count": 0,
                    "repost_count": 0,
                    "url": link
                })

            time.sleep(random.uniform(1, 2))

        except Exception as e:
            print("采集失败：", e)

    df = pd.DataFrame(data)

    if not df.empty:
        df = df.drop_duplicates(
            subset=["title", "url"]
        )

        df = df[
            df["content"]
            .astype(str)
            .str.len()
            > 10
        ]

    return df


# =========================
# 4. 批量关键词采集 + 自动去重更新
# =========================

def update_realtime_data():
    keywords = [
        "小米汽车",
        "小米SU7",
        "小米汽车事故",
        "小米SU7投诉",
        "小米汽车刹车"
    ]

    all_dfs = []

    for kw in keywords:
        temp = crawl_baidu_news(
            keyword=kw,
            pages=1
        )

        print(f"{kw} 采集数量：{len(temp)}")

        all_dfs.append(temp)

    if len(all_dfs) == 0:
        print("本次未采集到数据")
        return pd.DataFrame()

    new_df = pd.concat(
        all_dfs,
        ignore_index=True
    )

    if os.path.exists(DATA_PATH):
        old_df = pd.read_csv(
            DATA_PATH,
            encoding="utf-8-sig"
        )
    else:
        old_df = pd.DataFrame()

    old_count = len(old_df)

    final_df = pd.concat(
        [old_df, new_df],
        ignore_index=True
    )

    if not final_df.empty:
        final_df = final_df.drop_duplicates(
            subset=["title", "url"]
        )

    os.makedirs(
        os.path.dirname(DATA_PATH),
        exist_ok=True
    )

    final_df.to_csv(
        DATA_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("原有数据量：", old_count)
    print("更新后数据量：", len(final_df))
    print("本次新增数量：", len(final_df) - old_count)
    print("保存路径：", DATA_PATH)

    return final_df


# =========================
# 5. 主程序入口
# =========================

if __name__ == "__main__":
    update_realtime_data()
