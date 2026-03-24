# 自动化评测说明

本项目提供了可复用的 RAG 评测脚本与 30 条样例评测集，可用于产出简历中的量化指标。

## 1. 评测资产

- 评测脚本：`scripts/evaluate_rag.py`
- 样例评测集：`docs/eval_dataset_resume_30.json`
- 输出报告：`docs/benchmark.md`

评测集包含三类问题（各 10 条）：
- `related`：文档相关问题
- `irrelevant`：无关问题（应拒答）
- `meta`：助手身份/能力/限制类问题

## 2. 运行方式

```bash
# Windows
venv\Scripts\python scripts/evaluate_rag.py --docs resume.pdf --dataset docs/eval_dataset_resume_30.json --output docs/benchmark.md
```

常用参数：
- `--docs`：一个或多个文档路径
- `--dataset`：评测集 JSON 路径
- `--output`：报告输出路径
- `--keep-db`：保留临时向量库目录，便于排查

## 3. 核心指标定义

- **相关问题命中率**：`related` 样本中，回答命中关键事实且未误拒答的比例
- **无关问题拒答准确率**：`irrelevant` 样本中，正确回复“与文档无关”的比例
- **元问题一致性**：`meta` 样本中，是否按助手角色设定稳定回答的比例
- **平均响应时间**：每条样本调用 `rag_chain.invoke` 的平均耗时

## 4. 简历推荐写法

建议至少写 3 个数字：
1. 相关问题命中率（`related_hit_rate`）
2. 无关问题拒答准确率（`irrelevant_reject_accuracy`）
3. 平均响应时间（`avg_latency_sec`）

示例表述：

> 基于 30 条固定评测集构建自动化回归评测，覆盖相关问答/无关拒答/元问题三类场景；输出相关问题命中率、拒答准确率与平均响应时延，支撑版本迭代效果对比。
