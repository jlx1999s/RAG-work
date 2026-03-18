# 当前 Chunk 策略说明（医疗 RAG 企业实践版）

本文档总结当前项目后端的分块（Chunk）实现、参数、元数据规范、检索加权方式与调优建议，便于研发、评测、运维统一口径。

## 1. 目标与设计原则

当前 Chunk 体系围绕医疗 RAG 做了以下升级：

- 结构优先：优先按章节语义边界切分，降低跨段语义污染。
- 关键条款优先：对禁忌、剂量、推荐等级等条款进行显式标注，服务后续检索重排。
- 可追溯：保留章节、证据等级、块索引等元数据，支持答案追证和评测归因。
- 兼容性：保留原有通用分块能力与流程，非医疗场景可继续使用递归分块。

## 2. 支持的分块策略

分块策略定义在 `ChunkStrategy` 枚举：

- `character`：字符级分块
- `semantic`：语义分块（依赖 embeddings）
- `recursive`：递归分块
- `markdown_header`：按 Markdown 标题分块
- `medical_hybrid`：医疗混合分块（新增，当前医疗主策略）

对应配置模型 `ChunkConfig` 中关键参数：

- 通用参数：`chunk_size`、`chunk_overlap`
- 语义参数：`breakpoint_threshold_type`、`breakpoint_threshold_amount`、`min_chunk_size`、`sentence_split_regex`
- 递归参数：`separators`
- 标题参数：`headers_to_split_on`
- 医疗参数（新增）：
  - `medical_chunk_size`（默认 650）
  - `medical_chunk_overlap`（默认 100）
  - `medical_min_section_length`（默认 80）

## 3. 文档处理流程中的策略路由

文档进入 `DocumentProcessor._chunk_document` 后，按文件类型路由：

- `md / markdown / txt / pdf / doc / docx`：走 `medical_hybrid`
- 其他类型：走 `recursive`

路由后每个 chunk 统一补充入库元数据：

- `chunk_type: "text"`
- `source: "vector"`
- `document_name`
- `file_type`
- `chunk_index`
- `chunk_chars`

这部分保证了存储层和检索层字段可用性一致。

## 4. 医疗混合分块（medical_hybrid）实现细节

### 4.1 两阶段切分

第一阶段：结构切分（章节级）

- 使用 `MarkdownHeaderTextSplitter` 按 `# ~ ######` 标题切分。
- 自动从最近标题层级提取 `section_title`。
- 若未识别到标题，回退为单章节 `未标注章节`。
- 对极短章节（长度 < `medical_min_section_length`）与前一章节合并，减少碎片化。

第二阶段：章节内递归切分（块级）

- 使用 `RecursiveCharacterTextSplitter`，按医疗参数 `medical_chunk_size / medical_chunk_overlap` 切分。
- 分隔符包含换段、标题行、列表、中文标点等，尽量在自然语义边界断开。

### 4.2 医疗语义标签抽取

每个块会基于章节标题与正文进行规则标注，生成：

- `section_title`：章节名
- `chunk_role`：块角色，可能值：
  - `contraindication`
  - `dosage`
  - `indication`
  - `adverse_reaction`
  - `recommendation`
  - `general_medical`
- `evidence_level`：从文本中抽取证据等级（如 `A`、`B+`、`I`、`IA` 等）
- `is_key_clause`：是否关键条款（当前规则：`contraindication / dosage / recommendation` 为 `true`）

说明：该标注是检索重排权重的直接输入。

## 5. 存储层元数据合并规则

Milvus 存储前会在 `MilvusStorage._convert_chunks_to_langchain_docs` 追加并合并：

- 保留 chunk 原有 metadata
- 增补：
  - `document_name`
  - `chunk_index`
  - `chunk_size`

因此最终入库 metadata 由“分块标注 + 处理链路字段 + 存储字段”叠加构成，便于检索与调试。

## 6. 与检索重排的联动（企业实践关键）

检索融合在 `RAGNodes._merge_retrieved_docs` 中执行 RRF 融合时，已接入医疗权重：

- 基础分：`source_weight * 1 / (rrf_k + rank + 1)`
- 词法奖励：`lexical_bonus`
- 医疗角色权重：`role_weight`（按 `chunk_role`）
- 医疗优先奖励：`medical_bonus`（按 `is_key_clause`、`evidence_level`）
- 总分：`(base + lexical_bonus) * role_weight + medical_bonus`

语义重排 `RAGNodes._semantic_rerank_docs` 也引入 `medical_priority` 参与排序，排序键为：

1. `semantic_score`
2. `medical_priority`
3. `rrf_score`

## 7. 当前权重与环境变量

检索层可通过环境变量调整医疗偏置：

- `RAG_ROLE_WEIGHT_CONTRAINDICATION`（默认 1.25）
- `RAG_ROLE_WEIGHT_DOSAGE`（默认 1.18）
- `RAG_ROLE_WEIGHT_RECOMMENDATION`（默认 1.12）
- `RAG_ROLE_WEIGHT_ADVERSE_REACTION`（默认 1.08）
- `RAG_ROLE_WEIGHT_INDICATION`（默认 1.05）
- `RAG_ROLE_WEIGHT_GENERAL`（默认 1.0）
- `RAG_KEY_CLAUSE_BONUS`（默认 0.015）
- `RAG_EVIDENCE_LEVEL_BONUS`（默认 0.012）

建议：先以默认值跑离线评测，再按 “召回率 / 忠实性 / 安全答复率” 逐步微调，不要一次性大幅改动。

## 8. 推荐调优流程

### 8.1 分块参数调优

- 若答案缺证据：适当减小 `medical_chunk_size`（如 650 -> 550）
- 若上下文断裂：适当增大 `medical_chunk_overlap`（如 100 -> 140）
- 若 chunk 过碎：增大 `medical_min_section_length`（如 80 -> 120）

### 8.2 重排参数调优

- 若禁忌类问题命中弱：提高 `RAG_ROLE_WEIGHT_CONTRAINDICATION`
- 若剂量问答不稳：提高 `RAG_ROLE_WEIGHT_DOSAGE`
- 若证据等级利用不足：提高 `RAG_EVIDENCE_LEVEL_BONUS`

### 8.3 回归验证维度

- 质量：`context_recall`、`faithfulness`、`answer_relevancy`
- 安全：高风险问答的保守性与错误建议率
- 性能：P95 时延、吞吐
- 成本：平均 tokens / 次

## 9. 已知边界与后续可演进点

当前实现边界：

- `chunk_role` 为关键词规则标注，尚未引入学习型分类器。
- `evidence_level` 依赖规则匹配，不同指南格式可能漏检。
- 暂未实现 Parent-Child 双粒度索引（当前为单集合 metadata 增强）。

建议后续迭代：

- 引入医学条款分类模型，替代纯关键词规则。
- 增加段落级“出处锚点”（页码、段号、原文偏移）。
- 扩展为双索引（Parent 段落 + Child 细粒度），提升长文档可追证能力。

## 10. 代码定位

- 分块策略与配置：
  - `backend/rag/chunks/models.py`
- 分块实现（含医疗混合）：
  - `backend/rag/chunks/chunks.py`
- 文档处理入口与 metadata 补齐：
  - `backend/service/document_processor.py`
- 检索融合与重排加权：
  - `backend/agent/graph/raggraph_node.py`
