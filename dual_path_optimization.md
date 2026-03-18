# 双路径检索优化策略总结（企业实践落地版）

本文档总结当前项目在“双路径检索（向量 + 图谱）”上的优化设计、已落地能力、配置项、观测指标与调优建议，作为研发与运维统一参考。

## 1. 背景与目标

当前系统支持三种检索模式：

- `VECTOR_ONLY`
- `GRAPH_ONLY`
- `HYBRID`（向量+图谱融合）

双路径能提升复杂问答的覆盖能力，但天然会带来额外开销：

- 时延增加（多检索链路）
- 成本增加（更多调用与重排）
- 波动增加（图谱链路更易超时）

优化目标：

- 降低 P95 时延
- 降低图检索调用率与失败影响面
- 保持答案质量与可追证性

---

## 2. 总体优化架构

当前采用“路由治理 + 条件触发 + 并行执行 + 分层缓存 + 超时降级 + 熔断 + 动态预算 + 可观测”组合策略。

### 2.1 路由治理

- 修正 `AUTO` 分类下 `graph_only` 映射逻辑，确保进入 `GRAPH_ONLY`，避免误走 `HYBRID` 造成额外开销。

### 2.2 条件触发图检索

在 `HYBRID` 中不再无条件执行图检索，改为“满足条件时跳过图检索”：

- 向量候选数量足够
- 向量置信度达到阈值

只有当向量检索“覆盖不足或置信度不足”时才保留图检索。

### 2.3 并行执行

`HYBRID` 模式下，向量检索与图检索并发启动，减少总等待时间。

### 2.4 分层缓存（L1 + L2）

- L1：进程内 TTL 缓存（低延迟）
- L2：Redis 缓存（跨进程共享）

已覆盖：

- 向量检索结果缓存
- 图检索结果缓存

### 2.5 超时降级

图检索增加超时保护，超时后返回空图结果并继续流程，避免拖垮整体响应。

### 2.6 熔断机制

图检索连续失败达到阈值后，进入熔断窗口，窗口期内跳过图检索，保护系统稳定性。

### 2.7 动态预算

基于图检索延迟 EMA 自适应调整图检索预算（`max_docs`），在质量与时延间动态平衡。

### 2.8 可观测性

输出向量/图检索统计信息，支撑线上监控与离线调优。

---

## 3. 已落地能力清单

### 阶段一：基础降本

- 路由修正（`graph_only` 正确路由）
- 本地检索缓存
- 图检索超时降级

### 阶段二：效率提升

- `HYBRID` 并行化
- 图检索 Redis 二级缓存

### 阶段三：智能触发

- 向量置信度估计
- 置信度 + 数量联合判定是否跳图
- 检索统计字段补齐

### 阶段四：跨进程复用

- 向量检索 Redis 二级缓存接入

### 阶段五：稳定性增强

- 图检索熔断
- 图检索动态预算（EMA 驱动）

---

## 4. 关键流程说明

### 4.1 HYBRID 主流程

1. 计算图检索预算（动态）
2. 判断图熔断是否开启
3. 并行启动：
   - 向量检索（同步函数在线程执行）
   - 图检索（异步任务）
4. 获取向量结果并计算置信度
5. 满足“跳图条件”则取消图任务并直接融合向量结果
6. 否则等待图结果后做融合重排
7. 记录融合统计与检索统计

### 4.2 跳图条件

当以下条件同时满足时跳过图检索：

- `vector_docs >= max(conditional_graph_min_vector_docs, max_retrieval_docs)`
- `vector_confidence >= conditional_graph_confidence_threshold`

若熔断开启，也会强制跳图。

### 4.3 图检索稳定性逻辑

- 成功：清零连续失败计数，更新延迟 EMA
- 超时/异常：失败计数+1
- 达到阈值：打开熔断，进入冷却时间
- 冷却结束：自动恢复尝试

---

## 5. 缓存策略细节

### 5.1 缓存 Key 组成

统一由以下因子构成：

- 检索类型前缀（`vector` / `graph`）
- 归一化 query
- subquestions
- `max_docs`
- 额外上下文（例如 workspace）

### 5.2 缓存数据结构

- 向量缓存：
  - `retrieved_docs`
  - `vector_docs`
- 图缓存：
  - `docs`

文档序列化字段：

- `page_content`
- `metadata`

### 5.3 读写顺序

- 读：先本地缓存，未命中再 Redis，命中 Redis 回填本地
- 写：先本地缓存，再写 Redis（相同 TTL）

---

## 6. 检索统计字段（可观测）

### 6.1 向量检索统计 `vector_retrieval_stats`

- `cache_hit`
- `duration_ms`
- `question_count`
- `retrieved_count`
- `vector_candidate_count`
- `vector_confidence`

### 6.2 图检索统计 `graph_retrieval_stats`

- `cache_hit`
- `duration_ms`
- `retrieved_count`
- `timeout_seconds`
- `budget_docs`
- `timed_out`（可选）
- `error`（可选）

### 6.3 融合统计 `retrieval_fusion_stats`

- `vector_docs`
- `graph_docs`
- `merged_docs`
- `rrf_k`
- `mmr_lambda`
- `graph_skipped`
- `graph_circuit_open`
- `graph_budget_docs`
- `vector_confidence`
- `vector_confidence_threshold`
- `vector_selected_docs`

---

## 7. 关键环境变量

### 7.1 条件触发与融合

- `RAG_ENABLE_CONDITIONAL_GRAPH`（默认 `true`）
- `RAG_CONDITIONAL_GRAPH_MIN_VECTOR_DOCS`（默认 `3`）
- `RAG_CONDITIONAL_GRAPH_CONFIDENCE_THRESHOLD`（默认 `0.22`）
- `RAG_RRF_K`（默认 `60`）
- `RAG_MMR_LAMBDA`（默认 `0.75`）

### 7.2 缓存

- `RAG_RETRIEVAL_CACHE_ENABLED`（默认 `true`）
- `RAG_RETRIEVAL_CACHE_TTL_SECONDS`（默认 `300`）
- `RAG_RETRIEVAL_CACHE_MAX_ENTRIES`（默认 `512`）
- `RAG_REDIS_RETRIEVAL_CACHE_ENABLED`（默认 `true`）
- `RAG_REDIS_RETRIEVAL_CACHE_PREFIX`（默认 `rag:retrieval`）

### 7.3 图检索超时/熔断/预算

- `RAG_GRAPH_QUERY_TIMEOUT_SECONDS`（默认 `2.5`）
- `RAG_GRAPH_CIRCUIT_BREAKER_ENABLED`（默认 `true`）
- `RAG_GRAPH_CIRCUIT_FAIL_THRESHOLD`（默认 `3`）
- `RAG_GRAPH_CIRCUIT_COOLDOWN_SECONDS`（默认 `30`）
- `RAG_GRAPH_LATENCY_TARGET_MS`（默认 `600`）
- `RAG_GRAPH_BUDGET_MIN_DOCS`（默认 `1`）
- `RAG_GRAPH_BUDGET_MAX_DOCS`（默认 `8`）
- `RAG_GRAPH_LATENCY_EMA_ALPHA`（默认 `0.25`）

---

## 8. 推荐调优顺序（线上）

1. 先固定默认阈值，观察一周基础指标
2. 调 `RAG_CONDITIONAL_GRAPH_CONFIDENCE_THRESHOLD` 降图调用率
3. 调 `RAG_GRAPH_QUERY_TIMEOUT_SECONDS` 与 `RAG_GRAPH_CIRCUIT_*` 稳定 P95
4. 调 `RAG_GRAPH_LATENCY_TARGET_MS` 与 `RAG_GRAPH_BUDGET_*` 做质量-时延平衡
5. 结合业务问题类型，校准角色权重与融合权重

---

## 9. 验证建议

### 9.1 核心指标

- 性能：P50/P95 时延、超时率、图检索调用率
- 稳定：熔断触发率、降级率、错误率
- 质量：context recall、faithfulness、answer relevancy
- 成本：单问平均 tokens、检索调用次数

### 9.2 对照实验

- A 组：仅向量
- B 组：当前优化后双路径

重点观察：

- 复杂问题覆盖提升幅度
- 时延与成本上升是否在可接受区间

---

## 10. 代码位置

- 双路径检索主逻辑与优化实现：
  - `rag-backend/backend/agent/graph/raggraph_node.py`
- 图构建与节点编排：
  - `rag-backend/backend/agent/graph/raggraph.py`
- Redis 客户端：
  - `rag-backend/backend/config/redis.py`

---

## 11. 运维快速执行手册（1页）

以下给出三套可直接落地的参数模板，按业务目标选择：

- 保守：优先质量，允许较高开销
- 平衡：质量/时延折中，推荐默认线上起步
- 激进：优先时延与成本，接受一定召回下降

### 11.1 参数模板

#### A. 保守（质量优先）

- `RAG_ENABLE_CONDITIONAL_GRAPH=true`
- `RAG_CONDITIONAL_GRAPH_MIN_VECTOR_DOCS=5`
- `RAG_CONDITIONAL_GRAPH_CONFIDENCE_THRESHOLD=0.30`
- `RAG_GRAPH_QUERY_TIMEOUT_SECONDS=3.5`
- `RAG_GRAPH_CIRCUIT_FAIL_THRESHOLD=5`
- `RAG_GRAPH_CIRCUIT_COOLDOWN_SECONDS=20`
- `RAG_GRAPH_LATENCY_TARGET_MS=900`
- `RAG_GRAPH_BUDGET_MIN_DOCS=2`
- `RAG_GRAPH_BUDGET_MAX_DOCS=10`
- `RAG_RETRIEVAL_CACHE_TTL_SECONDS=300`

适用场景：

- 医疗高风险问答、可容忍更高时延

#### B. 平衡（推荐默认）

- `RAG_ENABLE_CONDITIONAL_GRAPH=true`
- `RAG_CONDITIONAL_GRAPH_MIN_VECTOR_DOCS=3`
- `RAG_CONDITIONAL_GRAPH_CONFIDENCE_THRESHOLD=0.22`
- `RAG_GRAPH_QUERY_TIMEOUT_SECONDS=2.5`
- `RAG_GRAPH_CIRCUIT_FAIL_THRESHOLD=3`
- `RAG_GRAPH_CIRCUIT_COOLDOWN_SECONDS=30`
- `RAG_GRAPH_LATENCY_TARGET_MS=600`
- `RAG_GRAPH_BUDGET_MIN_DOCS=1`
- `RAG_GRAPH_BUDGET_MAX_DOCS=8`
- `RAG_RETRIEVAL_CACHE_TTL_SECONDS=300`

适用场景：

- 大多数生产环境，追求稳态表现

#### C. 激进（性能优先）

- `RAG_ENABLE_CONDITIONAL_GRAPH=true`
- `RAG_CONDITIONAL_GRAPH_MIN_VECTOR_DOCS=2`
- `RAG_CONDITIONAL_GRAPH_CONFIDENCE_THRESHOLD=0.16`
- `RAG_GRAPH_QUERY_TIMEOUT_SECONDS=1.5`
- `RAG_GRAPH_CIRCUIT_FAIL_THRESHOLD=2`
- `RAG_GRAPH_CIRCUIT_COOLDOWN_SECONDS=45`
- `RAG_GRAPH_LATENCY_TARGET_MS=450`
- `RAG_GRAPH_BUDGET_MIN_DOCS=1`
- `RAG_GRAPH_BUDGET_MAX_DOCS=5`
- `RAG_RETRIEVAL_CACHE_TTL_SECONDS=600`

适用场景：

- 高并发成本敏感场景，对极难问题召回下降更容忍

### 11.2 上线步骤

1. 选择模板并在灰度环境生效
2. 连续观测 24h：
   - `P95 时延`
   - `graph_skipped 占比`
   - `graph_circuit_open 占比`
   - `timed_out 占比`
   - `answer faithfulness`
3. 若质量下降明显：
   - 提高 `RAG_CONDITIONAL_GRAPH_CONFIDENCE_THRESHOLD`
   - 增加 `RAG_GRAPH_QUERY_TIMEOUT_SECONDS`
   - 提高 `RAG_GRAPH_BUDGET_MAX_DOCS`
4. 若时延仍高：
   - 降低 `RAG_GRAPH_QUERY_TIMEOUT_SECONDS`
   - 降低 `RAG_GRAPH_BUDGET_MAX_DOCS`
   - 降低 `RAG_CONDITIONAL_GRAPH_MIN_VECTOR_DOCS`
5. 全量发布并保留回滚配置快照

### 11.3 回滚策略

出现以下任一情况建议立即回滚到“平衡模板”：

- `P95` 连续 15 分钟超目标阈值
- `timed_out` 或 `graph_circuit_open` 明显激增
- 人工评审发现高风险问答质量下滑

最小回滚动作：

- `RAG_CONDITIONAL_GRAPH_CONFIDENCE_THRESHOLD` 回调至 `0.22`
- `RAG_GRAPH_QUERY_TIMEOUT_SECONDS` 回调至 `2.5`
- `RAG_GRAPH_BUDGET_MAX_DOCS` 回调至 `8`
- `RAG_GRAPH_CIRCUIT_FAIL_THRESHOLD` 回调至 `3`
