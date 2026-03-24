"""
RAG 评测脚本

功能：
1. 从指定文档构建临时向量库
2. 执行固定评测集（related / irrelevant / meta）
3. 输出关键指标和失败样例
4. 生成 markdown 报告（默认 docs/benchmark.md）
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rag_agent.config import config
from rag_agent.core import DocumentLoader, DocumentSplitter, VectorStore, RAGChain
from rag_agent.utils.logger import logger


IRRELEVANT_SENTENCE = "与文档无关"


@dataclass
class EvalCaseResult:
    case_id: str
    case_type: str
    question: str
    passed: bool
    reason: str
    latency_sec: float
    answer_preview: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 自动评测脚本")
    parser.add_argument(
        "--docs",
        nargs="+",
        default=["resume.text"],
        help="用于评测的文档路径列表（默认: resume.text）",
    )
    parser.add_argument(
        "--dataset",
        default="docs/eval_dataset_resume_30.json",
        help="评测集 JSON 路径",
    )
    parser.add_argument(
        "--output",
        default="docs/benchmark.md",
        help="评测报告输出路径（markdown）",
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="是否保留临时向量库目录（默认清理）",
    )
    return parser.parse_args()


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"评测集不存在: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("评测集格式错误：应为非空列表")
    return data


def build_temp_vectorstore(doc_paths: List[str]) -> Tuple[VectorStore, str]:
    """构建临时向量库并返回 (vectorstore, temp_dir)"""
    config.validate()

    all_documents = []
    for doc_path in doc_paths:
        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"文档不存在: {doc_path}")
        docs = DocumentLoader.load(doc_path)
        filename = os.path.basename(doc_path)
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["filename"] = filename
        all_documents.extend(docs)

    splitter = DocumentSplitter()
    splits = splitter.split(all_documents)
    if not splits:
        raise RuntimeError("文档分块为空，无法评测")

    temp_dir = os.path.join(config.CHROMA_TEMP_DIR, "eval", str(uuid.uuid4()))
    os.makedirs(temp_dir, exist_ok=True)
    vectorstore = VectorStore(persist_directory=temp_dir)
    vectorstore.create_from_documents(splits, clear_existing=False)
    return vectorstore, temp_dir


def evaluate_case(case: Dict[str, Any], chain: RAGChain) -> EvalCaseResult:
    question = case["question"]
    case_type = case["type"]
    case_id = case.get("id", "UNKNOWN")
    target_filename = case.get("target_filename")

    start = time.perf_counter()
    resp = chain.invoke(question, target_filename=target_filename)
    latency = time.perf_counter() - start

    answer = (resp.answer or "").strip()
    is_refusal = IRRELEVANT_SENTENCE in answer

    passed = False
    reason = ""

    if case_type == "related":
        if is_refusal:
            passed = False
            reason = "被误判为无关问题"
        else:
            expected_all = case.get("expected_all", [])
            expected_any = case.get("expected_any", [])
            expected_contains = case.get("expected_contains")

            all_ok = all(k in answer for k in expected_all) if expected_all else True
            any_ok = any(k in answer for k in expected_any) if expected_any else True
            contain_ok = (expected_contains in answer) if expected_contains else True
            passed = all_ok and any_ok and contain_ok
            if not passed:
                reason = "回答未命中预期关键信息"

    elif case_type == "irrelevant":
        passed = is_refusal
        if not passed:
            reason = "无关问题未正确拒答"

    elif case_type == "meta":
        expected_contains = case.get("expected_contains", "我是小A")
        passed = expected_contains in answer
        if not passed:
            reason = "元问题回答不符合预期"
    else:
        passed = False
        reason = f"未知 case 类型: {case_type}"

    answer_preview = answer[:120] + ("..." if len(answer) > 120 else "")
    return EvalCaseResult(
        case_id=case_id,
        case_type=case_type,
        question=question,
        passed=passed,
        reason=reason,
        latency_sec=latency,
        answer_preview=answer_preview,
    )


def calculate_metrics(results: List[EvalCaseResult]) -> Dict[str, Any]:
    total = len(results)
    related = [r for r in results if r.case_type == "related"]
    irrelevant = [r for r in results if r.case_type == "irrelevant"]
    meta = [r for r in results if r.case_type == "meta"]

    def acc(items: List[EvalCaseResult]) -> float:
        if not items:
            return 0.0
        return sum(1 for i in items if i.passed) / len(items)

    metrics = {
        "total_cases": total,
        "overall_accuracy": acc(results),
        "related_hit_rate": acc(related),
        "irrelevant_reject_accuracy": acc(irrelevant),
        "meta_consistency": acc(meta),
        "avg_latency_sec": sum(r.latency_sec for r in results) / total if total else 0.0,
    }
    return metrics


def write_markdown_report(
    output_path: str,
    dataset_path: str,
    doc_paths: List[str],
    metrics: Dict[str, Any],
    results: List[EvalCaseResult],
) -> None:
    failed = [r for r in results if not r.passed]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []
    lines.append("# RAG 评测报告")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 评测集：`{dataset_path}`")
    lines.append(f"- 文档：`{', '.join(doc_paths)}`")
    lines.append("")
    lines.append("## 核心指标")
    lines.append("")
    lines.append(f"- 总样本数：**{metrics['total_cases']}**")
    lines.append(f"- 总体准确率：**{metrics['overall_accuracy'] * 100:.2f}%**")
    lines.append(f"- 相关问题命中率：**{metrics['related_hit_rate'] * 100:.2f}%**")
    lines.append(f"- 无关问题拒答准确率：**{metrics['irrelevant_reject_accuracy'] * 100:.2f}%**")
    lines.append(f"- 元问题一致性：**{metrics['meta_consistency'] * 100:.2f}%**")
    lines.append(f"- 平均响应时间：**{metrics['avg_latency_sec']:.3f}s**")
    lines.append("")
    lines.append("## 失败样例")
    lines.append("")

    if not failed:
        lines.append("无失败样例。")
    else:
        lines.append("| ID | 类型 | 问题 | 失败原因 | 回答片段 |")
        lines.append("|---|---|---|---|---|")
        for item in failed:
            q = item.question.replace("|", "\\|")
            rsn = item.reason.replace("|", "\\|")
            ans = item.answer_preview.replace("|", "\\|")
            lines.append(f"| {item.case_id} | {item.case_type} | {q} | {rsn} | {ans} |")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    args = parse_args()
    logger.info("开始执行 RAG 评测")

    dataset = load_dataset(args.dataset)
    vectorstore, temp_dir = build_temp_vectorstore(args.docs)
    chain = RAGChain(vectorstore)

    results: List[EvalCaseResult] = []
    for case in dataset:
        result = evaluate_case(case, chain)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        logger.info(
            f"{status} {result.case_id} [{result.case_type}] {result.question} | "
            f"{result.latency_sec:.3f}s"
        )

    metrics = calculate_metrics(results)
    write_markdown_report(args.output, args.dataset, args.docs, metrics, results)

    logger.info("评测完成")
    logger.info(f"总体准确率: {metrics['overall_accuracy'] * 100:.2f}%")
    logger.info(f"相关问题命中率: {metrics['related_hit_rate'] * 100:.2f}%")
    logger.info(f"无关问题拒答准确率: {metrics['irrelevant_reject_accuracy'] * 100:.2f}%")
    logger.info(f"元问题一致性: {metrics['meta_consistency'] * 100:.2f}%")
    logger.info(f"平均响应时间: {metrics['avg_latency_sec']:.3f}s")
    logger.info(f"报告输出: {args.output}")

    if not args.keep_db:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"已清理临时向量库: {temp_dir}")
    else:
        logger.info(f"保留临时向量库: {temp_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
