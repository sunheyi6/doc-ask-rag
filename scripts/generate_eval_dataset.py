"""
基于真实文档自动生成评测集（30条）

规则：
1. related: 由 LLM 基于文档内容生成（默认10条）
2. irrelevant: 固定模板（10条）
3. meta: 固定模板（10条）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rag_agent.core import DocumentLoader, DocumentSplitter
from rag_agent.core.llm import LLM


IRRELEVANT_CASES = [
    "今天北京天气怎么样？",
    "现在几点了？",
    "美国总统是谁？",
    "帮我写一首七言绝句",
    "推荐三家上海好吃的火锅店",
    "比特币今天多少钱？",
    "地球到月球大约多远？",
    "你会算 123 * 456 吗？",
    "请给我讲一下黑洞是什么",
    "帮我规划一趟去日本的旅游行程",
]

META_CASES = [
    "你是谁",
    "你是谁呀",
    "你叫什么名字",
    "你叫什么名字啊",
    "你能做什么",
    "你能帮我做什么呀",
    "你有什么限制",
    "你不会什么",
    "你好，你是谁",
    "自我介绍一下吧",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于真实文档生成评测集")
    parser.add_argument(
        "--docs",
        nargs="+",
        default=["uploaded_docs/resume.pdf"],
        help="文档路径列表",
    )
    parser.add_argument(
        "--output",
        default="docs/eval_dataset_real_30.json",
        help="输出评测集路径",
    )
    parser.add_argument(
        "--related-count",
        type=int,
        default=10,
        help="related 样本数量",
    )
    return parser.parse_args()


def _extract_json(text: str) -> List[Dict[str, Any]]:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    # 优先直接解析
    try:
        return json.loads(text)
    except Exception:
        pass

    # 回退：提取第一个完整 JSON 数组
    start = text.find("[")
    if start == -1:
        raise ValueError("未找到 JSON 数组起始符 '['")

    depth = 0
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        raise ValueError("未找到完整 JSON 数组结束符 ']'")

    return json.loads(text[start : end + 1])


def build_related_cases(docs: List[str], related_count: int) -> List[Dict[str, Any]]:
    all_docs = []
    filenames = []
    for p in docs:
        loaded = DocumentLoader.load(p)
        filename = os.path.basename(p)
        filenames.append(filename)
        for d in loaded:
            d.metadata = d.metadata or {}
            d.metadata["filename"] = filename
        all_docs.extend(loaded)

    splitter = DocumentSplitter(chunk_size=500, chunk_overlap=80)
    splits = splitter.split(all_docs)
    context = "\n\n---\n\n".join([s.page_content[:300] for s in splits[:20]])

    prompt = f"""
你是评测数据生成器。请根据提供的文档内容，生成 {related_count} 条“文档相关问题”评测样本。

要求：
1) 输出必须是 JSON 数组，不能有其他文本。
2) 每条格式：
{{
  "id": "R01",
  "type": "related",
  "question": "...",
  "target_filename": "{filenames[0]}",
  "expected_any": ["关键词1", "关键词2"]
}}
3) id 从 R01 开始递增。
4) question 必须是用户真实会问的问题。
5) expected_any 放可用于判定回答命中的关键词（1-3个）。
6) 仅基于文档内容，不要编造。

文档内容片段：
{context}
""".strip()

    llm = LLM()
    raw = llm.invoke(prompt)
    items = _extract_json(raw)

    normalized = []
    for idx, item in enumerate(items[:related_count], 1):
        normalized.append(
            {
                "id": f"R{idx:02d}",
                "type": "related",
                "question": str(item.get("question", "")).strip(),
                "target_filename": str(item.get("target_filename") or filenames[0]),
                "expected_any": item.get("expected_any", []) or item.get("expected_all", []),
            }
        )
    return normalized


def main() -> int:
    args = parse_args()

    related = build_related_cases(args.docs, args.related_count)

    irrelevant = [
        {
            "id": f"I{idx:02d}",
            "type": "irrelevant",
            "question": q,
            "target_filename": os.path.basename(args.docs[0]),
        }
        for idx, q in enumerate(IRRELEVANT_CASES, 1)
    ]

    meta = []
    for idx, q in enumerate(META_CASES, 1):
        expected = "作为小A" if q in {"你有什么限制", "你不会什么"} else "我是小A"
        meta.append(
            {
                "id": f"M{idx:02d}",
                "type": "meta",
                "question": q,
                "expected_contains": expected,
            }
        )

    dataset = related + irrelevant + meta
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Generated dataset: {args.output}")
    print(f"Total cases: {len(dataset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
