<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import BaseButton from '@/components/BaseButton.vue'
import BaseCard from '@/components/BaseCard.vue'
import { knowledgeAPI } from '@/api/knowledge.js'
import {
  cancelRagEvaluation,
  getEvalDatasetContent,
  getRagEvaluationStatus,
  getEvalHistory,
  getEvalHistoryItems,
  listEvalHistory,
  listEvalDatasets,
  saveEvalDataset,
  startRagEvaluationAsync,
  trainIntentClassifier
} from '@/api/evaluation.js'

const router = useRouter()
const route = useRoute()

const sampleDataset = `{"question":"RAG系统的主要目标是什么？","reference":"RAG系统的主要目标是结合检索与生成，在回答中利用外部知识以提高准确性与可解释性。"}
{"question":"向量检索通常使用什么表示文本？","reference":"向量检索通常使用文本的向量表示，例如由嵌入模型生成的稠密向量。"}
{"question":"图检索更擅长处理哪类信息？","reference":"图检索更擅长处理实体关系与结构化关联信息。"}`

const datasetText = ref(sampleDataset)
const limit = ref(10)
const evalScope = ref('rag_full')
const runPreset = ref('standard')
const retrievalMode = ref('vector_only')
const maxDocs = ref(3)
const workspace = ref('eval_ws')
const collectionId = ref('')
const runTag = ref('')
const enableRagas = ref(true)
const ragasLimit = ref(10)
const includeItemDetails = ref(true)
const cacheEnabled = ref(false)
const cacheNamespace = ref('eval-default')
const classifierModelPath = ref('')
const classifierPositiveThreshold = ref(0.75)
const classifierNegativeThreshold = ref(0.25)
const ciMode = ref(false)
const failOnGate = ref(false)
const baselineHistoryId = ref('')
const libraries = ref([])
const datasets = ref([])
const selectedDataset = ref('sample')
const datasetLoading = ref(false)
const loading = ref(false)
const saving = ref(false)
const result = ref(null)
const taskId = ref('')
const polling = ref(false)
const canceling = ref(false)
const datasetName = ref('')
const allowOverwrite = ref(false)
const histories = ref([])
const historyLoading = ref(false)
const historyDetailLoading = ref(false)
const historyDetail = ref(null)
const selectedHistoryId = ref('')
const historyItems = ref([])
const historyItemsLoading = ref(false)
const historyItemsPage = ref(1)
const historyItemsPageSize = ref(5)
const historyItemsTotal = ref(0)
const historyItemsStats = ref(null)
const activeLabTab = ref('classifier')
const classifierTrainLoading = ref(false)
const classifierTrainResult = ref(null)
const classifierTrainModelName = ref('')
const classifierTrainSplitStrategy = ref('random')
const classifierTrainValidRatio = ref(0.2)
const classifierTrainTestRatio = ref(0.1)
const classifierTrainSmoothing = ref(1.0)
const classifierTrainMinTokenFreq = ref(2)
const classifierTrainUseInlineEval = ref(false)
const labSampleMode = ref('all')
const labSampleKeyword = ref('')
const labSampleLimit = ref(12)
const progressEvents = ref([])
const taskSubmittedAt = ref(null)
const taskStartedAt = ref(null)
const taskRuntimeSeconds = ref(null)
const taskQueueWaitSeconds = ref(null)
const evaluationProgress = ref({
  stage: 'queued',
  percent: 0,
  processed: 0,
  total: 0,
  message: '',
  lab: null
})
let pollTimer = null

const datasetCount = computed(() => {
  return datasetText.value.split('\n').filter((line) => line.trim()).length
})

const metricDefinitions = [
  { key: 'context_precision', label: '上下文精度', tone: 'emerald' },
  { key: 'context_recall', label: '上下文召回', tone: 'cyan' },
  { key: 'answer_relevancy', label: '答案相关性', tone: 'violet' },
  { key: 'faithfulness', label: '事实一致性', tone: 'amber' }
]

const evalScopeOptions = [
  {
    value: 'rag_full',
    title: 'RAG 全链路',
    subtitle: 'Classifier + Retrieval + Generation + Medical Safety',
    description: '用于回归验收与发布门禁，适合正式版本评测'
  },
  {
    value: 'classifier_only',
    title: 'Classifier Lab',
    subtitle: '意图分类训练/验证',
    description: '验证 need_retrieval 分类器效果与阈值'
  },
  {
    value: 'retrieval_only',
    title: 'Retrieval Lab',
    subtitle: '检索链路诊断',
    description: '重点看 Hit@k、Recall@k、Context 命中'
  },
  {
    value: 'generation_only',
    title: 'Generation Lab',
    subtitle: '生成质量验证',
    description: '重点看 Faithfulness、Completeness、Citation'
  },
  {
    value: 'medical_safety_only',
    title: 'Medical Safety Lab',
    subtitle: '医疗 SOP 与红线',
    description: '重点看红线召回、误拦截与转人工准确性'
  }
]

const routeLabToScopeMap = {
  'rag-full': 'rag_full',
  classifier: 'classifier_only',
  retrieval: 'retrieval_only',
  generation: 'generation_only',
  'medical-safety': 'medical_safety_only'
}

const scopeToRouteLabMap = {
  rag_full: 'rag-full',
  classifier_only: 'classifier',
  retrieval_only: 'retrieval',
  generation_only: 'generation',
  medical_safety_only: 'medical-safety'
}

const formatScore = (value) => {
  if (value === null || value === undefined) return '--'
  const num = Number(value)
  if (Number.isNaN(num)) return '--'
  return num.toFixed(3)
}

const metricValue = (key) => {
  return result.value?.run?.quality_summary?.[key] ?? result.value?.summary?.[key]
}

const getHistorySummaryValue = (item, key) => {
  return item?.result?.run?.quality_summary?.[key] ?? item?.result?.summary?.[key] ?? null
}

const getHistoryTotal = (item) => {
  const runTotal = item?.result?.run?.dataset?.used_rows
  if (runTotal !== undefined && runTotal !== null) {
    return runTotal
  }
  if (item && item.result && item.result.total !== undefined && item.result.total !== null) {
    return item.result.total
  }
  if (item && item.dataset_used_lines !== undefined && item.dataset_used_lines !== null) {
    return item.dataset_used_lines
  }
  return 0
}

const getHistoryElapsed = (item) => {
  const elapsed = item?.result?.run?.performance_summary?.run_elapsed_ms
  if (elapsed !== undefined && elapsed !== null) {
    return elapsed
  }
  if (item && item.result && item.result.elapsed_ms !== undefined && item.result.elapsed_ms !== null) {
    return item.result.elapsed_ms
  }
  return 0
}

const getItemMetricValue = (item, key) => {
  if (!item || !item.metrics) return null
  return item.metrics[key]
}

const resultTotal = computed(() => {
  return result.value?.run?.dataset?.used_rows ?? result.value?.total ?? 0
})

const resultElapsedMs = computed(() => {
  return result.value?.run?.performance_summary?.run_elapsed_ms ?? result.value?.elapsed_ms ?? 0
})

const resultPerformance = computed(() => {
  return result.value?.run?.performance_summary ?? result.value?.performance_summary ?? null
})

const resultRetrieval = computed(() => {
  return result.value?.run?.retrieval_summary ?? result.value?.retrieval_summary ?? null
})

const resultRouting = computed(() => {
  return result.value?.run?.routing_summary ?? result.value?.routing_summary ?? null
})

const resultStability = computed(() => {
  return result.value?.run?.stability_summary ?? result.value?.stability_summary ?? null
})

const resultSop = computed(() => {
  return result.value?.run?.sop_summary ?? result.value?.sop_summary ?? null
})

const resultAnswerOverlap = computed(() => {
  return result.value?.run?.answer_overlap_summary ?? result.value?.answer_overlap_summary ?? null
})

const resultSlices = computed(() => {
  return result.value?.run?.slice_summary ?? result.value?.slice_summary ?? null
})

const resultQualityGates = computed(() => {
  return result.value?.run?.quality_gate_summary ?? result.value?.quality_gate_summary ?? null
})

const resultReleaseDecision = computed(() => {
  return result.value?.run?.release_decision ?? result.value?.release_decision ?? null
})

const resultBaseline = computed(() => {
  return result.value?.run?.baseline_comparison ?? result.value?.baseline_comparison ?? null
})

const resultCost = computed(() => {
  return result.value?.run?.cost_summary ?? result.value?.cost_summary ?? null
})

const resultCache = computed(() => {
  return result.value?.run?.cache_summary ?? result.value?.cache_summary ?? null
})

const resultBadcase = computed(() => {
  return result.value?.run?.badcase_summary ?? result.value?.badcase_summary ?? null
})

const resultReproducibility = computed(() => {
  return result.value?.run?.reproducibility_summary ?? result.value?.reproducibility_summary ?? null
})

const resultClassifier = computed(() => {
  return result.value?.run?.classifier_summary ?? result.value?.classifier_summary ?? null
})

const resultClassifierGate = computed(() => {
  return result.value?.run?.classifier_quality_gate_summary ?? result.value?.classifier_quality_gate_summary ?? null
})

const resultEvalLabs = computed(() => {
  return result.value?.run?.eval_labs ?? result.value?.eval_labs ?? null
})

const resultModuleLabs = computed(() => {
  return result.value?.run?.module_labs ?? result.value?.module_labs ?? null
})

const resultEvalScope = computed(() => {
  return (
    result.value?.run?.config?.eval_scope
    ?? result.value?.eval_scope
    ?? evalScope.value
  )
})

const resultIsClassifierOnly = computed(() => resultEvalScope.value === 'classifier_only')

const resultAllowedLabKeys = computed(() => {
  const scope = String(resultEvalScope.value || '')
  if (scope === 'classifier_only') return ['classifier']
  if (scope === 'retrieval_only') return ['retrieval']
  if (scope === 'generation_only') return ['generation']
  if (scope === 'medical_safety_only') return ['medical_safety']
  return ['classifier', 'retrieval', 'generation', 'medical_safety']
})

const moduleLabEntries = computed(() => {
  const labs = resultModuleLabs.value || {}
  const order = [
    { key: 'classifier', label: 'Classifier Lab' },
    { key: 'retrieval', label: 'Retrieval Lab' },
    { key: 'generation', label: 'Generation Lab' },
    { key: 'medical_safety', label: 'Medical Safety Lab' }
  ]
  const allowed = new Set(resultAllowedLabKeys.value)
  return order
    .map((item) => ({
      ...item,
      data: labs[item.key] || null
    }))
    .filter((item) => allowed.has(item.key) && item.data)
})

const activeModuleLab = computed(() => {
  const map = resultModuleLabs.value || {}
  return map[activeLabTab.value] || null
})

const resultItems = computed(() => {
  const items = result.value?.items
  return Array.isArray(items) ? items : []
})

const activeLabQuickMetrics = computed(() => {
  const lab = activeModuleLab.value || {}
  const metrics = lab?.metrics || {}
  if (activeLabTab.value === 'classifier') {
    return [
      { key: 'labeled_items', label: '标注样本', value: metrics?.labeled_items },
      { key: 'accuracy', label: '准确率', value: formatPercent(metrics?.accuracy) },
      { key: 'f1', label: 'F1', value: formatNumber(metrics?.f1, 3) },
      { key: 'uncertain_rate', label: '不确定率', value: formatPercent(metrics?.uncertain_rate) }
    ]
  }
  if (activeLabTab.value === 'retrieval') {
    return [
      { key: 'hit_at_3', label: 'Hit@3', value: formatPercent(metrics?.hit_at_k?.['3']) },
      { key: 'mrr', label: 'MRR', value: formatNumber(metrics?.mrr, 3) },
      { key: 'ndcg_at_3', label: 'nDCG@3', value: formatNumber(metrics?.ndcg_at_k?.['3'], 3) },
      { key: 'context_recall', label: 'Context Recall', value: formatPercent(metrics?.context_recall) }
    ]
  }
  if (activeLabTab.value === 'generation') {
    return [
      { key: 'faithfulness', label: 'Faithfulness', value: formatPercent(metrics?.faithfulness) },
      { key: 'completeness', label: 'Completeness', value: formatPercent(metrics?.completeness) },
      { key: 'citation_coverage', label: 'Citation Coverage', value: formatPercent(metrics?.citation_coverage) },
      { key: 'answer_relevancy', label: 'Answer Relevancy', value: formatPercent(metrics?.answer_relevancy) }
    ]
  }
  return [
    { key: 'redline_recall', label: '红线召回', value: formatPercent(metrics?.redline_recall) },
    { key: 'false_interception_rate', label: '误拦截率', value: formatPercent(metrics?.false_interception_rate) },
    { key: 'hallucination_rate', label: '幻觉率', value: formatPercent(metrics?.hallucination_rate) },
    { key: 'handoff_accuracy', label: '转人工准确率', value: formatPercent(metrics?.handoff_accuracy) }
  ]
})

const activeLabSampleModes = computed(() => {
  if (activeLabTab.value === 'classifier') {
    return [
      { value: 'all', label: '全部样本' },
      { value: 'mismatch', label: '标签错判' },
      { value: 'uncertain', label: '不确定样本' },
      { value: 'error', label: '执行错误' }
    ]
  }
  if (activeLabTab.value === 'retrieval') {
    return [
      { value: 'all', label: '全部样本' },
      { value: 'miss', label: '检索未命中' },
      { value: 'need', label: '需检索样本' },
      { value: 'error', label: '执行错误' }
    ]
  }
  if (activeLabTab.value === 'generation') {
    return [
      { value: 'all', label: '全部样本' },
      { value: 'unsupported', label: '证据不足' },
      { value: 'low_completeness', label: '低完整性' },
      { value: 'error', label: '执行错误' }
    ]
  }
  return [
    { value: 'all', label: '全部样本' },
    { value: 'handoff', label: '触发转人工' },
    { value: 'high_risk', label: '高风险/急诊' },
    { value: 'badcase', label: '安全Badcase' }
  ]
})

const activeLabSampleItems = computed(() => {
  const mode = String(labSampleMode.value || 'all')
  const keyword = String(labSampleKeyword.value || '').trim().toLowerCase()
  const maxCount = Math.max(1, Number(labSampleLimit.value) || 12)
  const filtered = resultItems.value.filter((item) => {
    const route = item?.routing || {}
    const sop = item?.sop || {}
    const labelEval = item?.classifier_label_eval || {}
    const retrievalEval = item?.retrieval_label_eval || {}
    const badcase = item?.badcase || {}
    const alignment = item?.generation_alignment || {}
    let passMode = true
    if (activeLabTab.value === 'classifier') {
      if (mode === 'mismatch') passMode = labelEval?.has_label && labelEval?.is_correct === false
      else if (mode === 'uncertain') passMode = route?.statistical_classifier?.decision === null
      else if (mode === 'error') passMode = item?.status === 'error'
    } else if (activeLabTab.value === 'retrieval') {
      if (mode === 'miss') passMode = retrievalEval?.context_hit === false
      else if (mode === 'need') passMode = route?.need_retrieval === true
      else if (mode === 'error') passMode = item?.status === 'error'
    } else if (activeLabTab.value === 'generation') {
      if (mode === 'unsupported') passMode = Number(alignment?.citation_coverage ?? 1) < 0.5
      else if (mode === 'low_completeness') passMode = Number(alignment?.completeness ?? 1) < 0.5
      else if (mode === 'error') passMode = item?.status === 'error'
    } else {
      if (mode === 'handoff') passMode = sop?.handoff_required === true
      else if (mode === 'high_risk') {
        const triage = String(sop?.triage_level || '').toLowerCase()
        passMode = triage === 'emergency' || triage === 'urgent'
      } else if (mode === 'badcase') {
        passMode = Boolean(badcase?.is_badcase) || Number(badcase?.risk_score ?? 0) >= 0.6
      }
    }
    if (!passMode) return false
    if (!keyword) return true
    const haystack = [
      item?.question,
      item?.reference,
      item?.answer,
      sop?.handoff_reason,
      route?.reason,
      ...(badcase?.tags || [])
    ]
      .filter(Boolean)
      .join(' ')
      .toLowerCase()
    return haystack.includes(keyword)
  })
  return filtered.slice(0, maxCount)
})

const historyDetailSummary = computed(() => {
  return historyDetail.value?.result?.run?.quality_summary ?? historyDetail.value?.result?.summary ?? null
})

const historyDetailStability = computed(() => {
  return historyDetail.value?.result?.run?.stability_summary ?? historyDetail.value?.result?.stability_summary ?? null
})

const historyDetailSop = computed(() => {
  return historyDetail.value?.result?.run?.sop_summary ?? historyDetail.value?.result?.sop_summary ?? null
})

const historyDetailRouting = computed(() => {
  return historyDetail.value?.result?.run?.routing_summary ?? historyDetail.value?.result?.routing_summary ?? null
})

const historyDetailGates = computed(() => {
  return historyDetail.value?.result?.run?.quality_gate_summary ?? historyDetail.value?.result?.quality_gate_summary ?? null
})

const historyDetailBaseline = computed(() => {
  return historyDetail.value?.result?.run?.baseline_comparison ?? historyDetail.value?.result?.baseline_comparison ?? null
})

const historyDetailClassifier = computed(() => {
  return historyDetail.value?.result?.run?.classifier_summary ?? historyDetail.value?.result?.classifier_summary ?? null
})

const historyDetailBadcase = computed(() => {
  return historyDetail.value?.result?.run?.badcase_summary ?? historyDetail.value?.result?.badcase_summary ?? null
})

const historyDetailReproducibility = computed(() => {
  return historyDetail.value?.result?.run?.reproducibility_summary ?? historyDetail.value?.result?.reproducibility_summary ?? null
})

const classifierTrainMetrics = computed(() => {
  return classifierTrainResult.value?.metrics || null
})

const isClassifierOnly = computed(() => evalScope.value === 'classifier_only')
const selectedScopeOption = computed(() => {
  return evalScopeOptions.find((item) => item.value === evalScope.value) || evalScopeOptions[0]
})
const supportsRagas = computed(() => ['rag_full', 'generation_only'].includes(evalScope.value))
const progressPercent = computed(() => {
  const num = Number(evaluationProgress.value?.percent ?? 0)
  if (Number.isNaN(num)) return 0
  return Math.max(0, Math.min(100, num))
})

const progressTimeline = computed(() => {
  const stage = String(evaluationProgress.value?.stage || '').toLowerCase()
  const stageOrder = isClassifierOnly.value
    ? ['queued', 'init', 'dataset_ready', 'running', 'summarizing', 'completed', 'canceled']
    : ['queued', 'init', 'dataset_ready', 'running', 'ragas', 'summarizing', 'completed', 'canceled']
  let currentIdx = stageOrder.indexOf(stage)
  if (currentIdx < 0 && (stage === 'failed' || stage === 'canceled')) {
    currentIdx = Number(evaluationProgress.value?.processed || 0) > 0 ? 3 : 0
  }
  const terminalLike = stage === 'failed' || stage === 'canceled'
  return stageOrder.map((key, idx) => ({
    key,
    label: formatProgressStage(key),
    status:
      terminalLike
        ? (idx <= Math.max(0, currentIdx) ? 'done' : 'todo')
        : idx < currentIdx
          ? 'done'
          : idx === currentIdx
            ? 'active'
            : 'todo'
  }))
})

const progressEventRows = computed(() => {
  const events = Array.isArray(progressEvents.value) ? progressEvents.value : []
  return events.slice(-12).reverse()
})

const currentStageElapsedSec = computed(() => {
  const stage = String(evaluationProgress.value?.stage || '').toLowerCase()
  if (!stage) return null
  const events = Array.isArray(progressEvents.value) ? progressEvents.value : []
  if (!events.length) return null
  let stageStartTs = null
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const eventStage = String(events[i]?.stage || '').toLowerCase()
    if (eventStage === stage) {
      stageStartTs = Number(events[i]?.ts || 0)
      continue
    }
    if (stageStartTs) break
  }
  if (!stageStartTs) return null
  const nowSec = Date.now() / 1000
  const elapsed = nowSec - stageStartTs
  if (!Number.isFinite(elapsed) || elapsed < 0) return null
  return elapsed
})

const progressEtaSec = computed(() => {
  const startedAt = Number(taskStartedAt.value || 0)
  const processed = Number(evaluationProgress.value?.processed || 0)
  const total = Number(evaluationProgress.value?.total || 0)
  if (!startedAt || !Number.isFinite(startedAt) || processed <= 0 || total <= 0 || processed >= total) {
    return null
  }
  const elapsed = Math.max(0, Date.now() / 1000 - startedAt)
  if (elapsed <= 0) return null
  const avgPerItem = elapsed / processed
  const remaining = avgPerItem * (total - processed)
  if (!Number.isFinite(remaining) || remaining < 0) return null
  return remaining
})

const historyItemsTotalPages = computed(() => {
  if (!historyItemsTotal.value || historyItemsTotal.value <= 0) return 1
  return Math.max(1, Math.ceil(historyItemsTotal.value / historyItemsPageSize.value))
})

const historyItemsAvailable = computed(() => {
  return historyItemsTotal.value > 0
})

const sourceEntries = computed(() => {
  const distribution = resultRetrieval.value?.source_distribution || {}
  return Object.entries(distribution)
})

const sampleLines = computed(() => {
  return sampleDataset.split('\n').filter((line) => line.trim()).length
})

const selectedDatasetInfo = computed(() => {
  if (selectedDataset.value === 'sample') {
    return {
      label: '内置示例',
      lines: sampleLines.value,
      size: null,
      source: 'sample'
    }
  }
  return datasets.value.find((item) => item.name === selectedDataset.value) || null
})

const datasetExists = computed(() => {
  const name = datasetName.value.trim()
  if (!name) return false
  const fileName = name.toLowerCase().endsWith('.jsonl') ? name : `${name}.jsonl`
  return datasets.value.some((item) => {
    const source = item?.source || 'user'
    return item.name === fileName && source === 'user'
  })
})

const workspaceOptions = computed(() => {
  return [{ value: 'eval_ws', label: '默认 eval_ws' }]
})

const formatBytes = (value) => {
  if (!value && value !== 0) return '--'
  const size = Number(value)
  if (Number.isNaN(size)) return '--'
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  return `${(size / (1024 * 1024)).toFixed(1)} MB`
}

const formatHistoryTime = (value) => {
  if (!value) return '--'
  const timeValue = Number(value)
  const timestamp = timeValue > 1000000000000 ? timeValue : timeValue * 1000
  return new Date(timestamp).toLocaleString('zh-CN', {
    year: 'numeric',
    month: 'numeric',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

const formatProgressTime = (value) => {
  if (!value) return '--'
  const ts = Number(value)
  if (!Number.isFinite(ts) || ts <= 0) return '--'
  return new Date(ts * 1000).toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  })
}

const formatNumber = (value, digits = 2) => {
  if (value === null || value === undefined) return '--'
  const num = Number(value)
  if (Number.isNaN(num)) return '--'
  return num.toFixed(digits)
}

const formatPercent = (value) => {
  if (value === null || value === undefined) return '--'
  const num = Number(value)
  if (Number.isNaN(num)) return '--'
  return `${(num * 100).toFixed(1)}%`
}

const formatDurationSec = (value) => {
  if (value === null || value === undefined) return '--'
  const num = Number(value)
  if (!Number.isFinite(num) || num < 0) return '--'
  if (num < 60) return `${num.toFixed(1)}s`
  const mins = Math.floor(num / 60)
  const secs = Math.round(num % 60)
  return `${mins}m ${secs}s`
}

const formatHashShort = (value) => {
  const text = String(value || '').trim()
  if (!text) return '--'
  if (text.length <= 12) return text
  return `${text.slice(0, 12)}...`
}

const labEnabledLabel = (lab) => {
  if (!lab) return '未知'
  return lab.enabled ? '已启用' : '已禁用'
}

const labEnabledClass = (lab) => {
  if (!lab) return 'bg-slate-50 text-slate-700 border-slate-200'
  if (lab.enabled) return 'bg-emerald-50 text-emerald-700 border-emerald-200'
  return 'bg-slate-50 text-slate-700 border-slate-200'
}

const defaultLabTabByScope = (scope) => {
  if (scope === 'classifier_only') return 'classifier'
  if (scope === 'retrieval_only') return 'retrieval'
  if (scope === 'generation_only') return 'generation'
  if (scope === 'medical_safety_only') return 'medical_safety'
  return 'retrieval'
}

const formatProgressStage = (stage) => {
  const key = String(stage || '').toLowerCase()
  if (key === 'queued') return '排队中'
  if (key === 'init') return '初始化'
  if (key === 'dataset_ready') return '数据就绪'
  if (key === 'running') return '执行中'
  if (key === 'ragas') return '语义评测'
  if (key === 'summarizing') return '汇总中'
  if (key === 'completed') return '已完成'
  if (key === 'canceled') return '已取消'
  if (key === 'failed') return '失败'
  return key || '--'
}

const progressStepClass = (status) => {
  if (status === 'done') return 'border-emerald-200 bg-emerald-50 text-emerald-700'
  if (status === 'active') return 'border-slate-300 bg-slate-900 text-white'
  return 'border-slate-200 bg-white text-slate-500'
}

const releaseActionLabel = (decision) => {
  const value = String(decision || '').toLowerCase()
  if (value === 'allow') return '允许发布'
  if (value === 'warn') return '告警放行'
  if (value === 'block') return '阻断发布'
  return '--'
}

const progressBadgeClass = (stage) => {
  const key = String(stage || '').toLowerCase()
  if (key === 'failed') return 'bg-rose-50 text-rose-700 border-rose-200'
  if (key === 'canceled') return 'bg-orange-50 text-orange-700 border-orange-200'
  if (key === 'completed') return 'bg-emerald-50 text-emerald-700 border-emerald-200'
  if (key === 'ragas') return 'bg-violet-50 text-violet-700 border-violet-200'
  if (key === 'running') return 'bg-cyan-50 text-cyan-700 border-cyan-200'
  if (key === 'summarizing') return 'bg-amber-50 text-amber-700 border-amber-200'
  return 'bg-slate-50 text-slate-700 border-slate-200'
}

const applyPreset = (preset) => {
  const key = String(preset || '').toLowerCase()
  if (key === 'smoke') {
    limit.value = 8
    ragasLimit.value = supportsRagas.value ? 5 : 0
    includeItemDetails.value = true
    return
  }
  if (key === 'full') {
    limit.value = 0
    ragasLimit.value = supportsRagas.value ? 0 : 0
    includeItemDetails.value = true
    return
  }
  limit.value = 25
  ragasLimit.value = supportsRagas.value ? 10 : 0
  includeItemDetails.value = true
}

const getItemSop = (item) => {
  return item?.sop || {}
}

const formatTriageLabel = (value) => {
  const triage = String(value || '').toLowerCase()
  if (triage === 'emergency') return '急诊'
  if (triage === 'urgent') return '紧急'
  if (triage === 'routine') return '常规'
  return '未知'
}

const formatBoolFlag = (value) => {
  if (value === null || value === undefined) return '--'
  return value ? '是' : '否'
}

const gateStatusLabel = (status) => {
  if (status === 'pass') return '通过'
  if (status === 'fail') return '失败'
  return '跳过'
}

const gateStatusClass = (status) => {
  if (status === 'pass') return 'bg-emerald-50 text-emerald-700 border-emerald-200'
  if (status === 'fail') return 'bg-rose-50 text-rose-700 border-rose-200'
  return 'bg-slate-50 text-slate-700 border-slate-200'
}

const formatDelta = (value) => {
  if (value === null || value === undefined) return '--'
  const num = Number(value)
  if (Number.isNaN(num)) return '--'
  const prefix = num > 0 ? '+' : ''
  return `${prefix}${num.toFixed(4)}`
}

const formatRoutingStageLabel = (stage) => {
  const normalized = String(stage || '').toLowerCase()
  if (normalized === 'rule') return '规则'
  if (normalized === 'statistical_classifier') return '统计分类器'
  if (normalized === 'lightweight_classifier') return '轻量分类器'
  if (normalized === 'llm') return 'LLM兜底'
  if (normalized === 'fallback_error') return '异常兜底'
  if (normalized === 'medical_handoff') return '医疗转人工短路'
  return normalized || '未知'
}

const routingStageClass = (stage) => {
  const normalized = String(stage || '').toLowerCase()
  if (normalized === 'rule') return 'bg-emerald-50 text-emerald-700 border-emerald-200'
  if (normalized === 'statistical_classifier') return 'bg-cyan-50 text-cyan-700 border-cyan-200'
  if (normalized === 'lightweight_classifier') return 'bg-sky-50 text-sky-700 border-sky-200'
  if (normalized === 'llm') return 'bg-amber-50 text-amber-700 border-amber-200'
  if (normalized === 'fallback_error') return 'bg-rose-50 text-rose-700 border-rose-200'
  return 'bg-slate-50 text-slate-700 border-slate-200'
}

const formatEvalScopeLabel = (scope) => {
  const value = String(scope || '').toLowerCase()
  if (value === 'rag_full') return '全链路'
  if (value === 'classifier_only') return 'Classifier'
  if (value === 'retrieval_only') return 'Retrieval'
  if (value === 'generation_only') return 'Generation'
  if (value === 'medical_safety_only') return 'Medical Safety'
  return value || '--'
}

const getHistoryEvalScope = (item) => {
  return item?.eval_scope || item?.result?.run?.config?.eval_scope || null
}

const getHistoryReleaseAction = (item) => {
  return item?.result?.run?.release_decision?.action || item?.result?.release_decision?.action || null
}

const releaseActionClass = (action) => {
  const value = String(action || '').toLowerCase()
  if (value === 'allow') return 'bg-emerald-50 text-emerald-700 border-emerald-200'
  if (value === 'warn') return 'bg-amber-50 text-amber-700 border-amber-200'
  if (value === 'block') return 'bg-rose-50 text-rose-700 border-rose-200'
  return 'bg-slate-50 text-slate-700 border-slate-200'
}

const resolveScopeFromRouteLab = (labParam) => {
  const key = String(labParam || '').trim().toLowerCase()
  return routeLabToScopeMap[key] || 'rag_full'
}

const resolveRouteLabFromScope = (scope) => {
  const key = String(scope || '').trim().toLowerCase()
  return scopeToRouteLabMap[key] || 'rag-full'
}

const syncRouteToScope = async (scope, replace = false) => {
  const targetLab = resolveRouteLabFromScope(scope)
  const currentLab = String(route.params.lab || '').trim().toLowerCase()
  if (currentLab === targetLab) return
  const targetRoute = { name: 'evaluation', params: { lab: targetLab } }
  if (replace) {
    await router.replace(targetRoute)
  } else {
    await router.push(targetRoute)
  }
}

const selectScope = (scope) => {
  if (!scope) return
  evalScope.value = scope
  syncRouteToScope(scope).catch(() => {})
}

const downloadTextFile = (filename, content) => {
  if (!content) return
  const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

const buildLabReportMarkdown = (payload, labKey, labTitle, exportItems = []) => {
  const run = payload?.run || {}
  const lab = run?.module_labs?.[labKey] || payload?.module_labs?.[labKey] || {}
  const metrics = lab?.metrics || lab?.summary || {}
  const qualityGate = lab?.quality_gate || {}
  const hardGate = lab?.hard_gate || {}
  const releaseDecision = run?.release_decision || payload?.release_decision || {}
  const lines = []
  lines.push(`# ${labTitle} 评测报告`)
  lines.push('')
  lines.push(`- 导出时间：${new Date().toLocaleString('zh-CN')}`)
  lines.push(`- Run ID：${run?.run_id || '--'}`)
  lines.push(`- 数据集：${run?.dataset?.name || payload?.dataset_name || '--'}`)
  lines.push(`- 样本数：${run?.dataset?.used_rows ?? payload?.total ?? '--'}`)
  lines.push(`- 发布决策：${releaseActionLabel(releaseDecision?.action)}`)
  lines.push('')
  lines.push('## 核心指标')
  lines.push('```json')
  lines.push(JSON.stringify(metrics, null, 2))
  lines.push('```')
  if (Object.keys(qualityGate || {}).length > 0) {
    lines.push('')
    lines.push('## 模块门禁')
    lines.push('```json')
    lines.push(JSON.stringify(qualityGate, null, 2))
    lines.push('```')
  }
  if (Object.keys(hardGate || {}).length > 0) {
    lines.push('')
    lines.push('## 安全硬门禁')
    lines.push('```json')
    lines.push(JSON.stringify(hardGate, null, 2))
    lines.push('```')
  }
  lines.push('')
  lines.push('## 样本摘录')
  if (!exportItems.length) {
    lines.push('- 无样本可导出')
  } else {
    exportItems.slice(0, 20).forEach((item, index) => {
      lines.push(`${index + 1}. Q: ${item?.question || '--'}`)
      lines.push(`A: ${item?.answer || '--'}`)
      lines.push(`状态: ${item?.status || '--'} | 路由: ${formatRoutingStageLabel(item?.routing?.stage)} | 上下文数: ${item?.contexts_count ?? 0}`)
      lines.push('')
    })
  }
  return lines.join('\n')
}

const exportActiveLabReport = () => {
  if (!result.value) {
    ElMessage.warning('暂无可导出的评测结果')
    return
  }
  const titles = {
    classifier: 'Classifier Lab',
    retrieval: 'Retrieval Lab',
    generation: 'Generation Lab',
    medical_safety: 'Medical Safety Lab'
  }
  const labKey = activeLabTab.value
  const markdown = buildLabReportMarkdown(
    result.value,
    labKey,
    titles[labKey] || labKey,
    activeLabSampleItems.value || []
  )
  const runId = result.value?.run?.run_id || `run_${Date.now()}`
  downloadTextFile(`eval_${runId}_${labKey}.md`, markdown)
  ElMessage.success('当前实验室报告已导出')
}

const exportFullRunReport = () => {
  if (!result.value) {
    ElMessage.warning('暂无可导出的评测结果')
    return
  }
  const sections = []
  const labs = [
    ['classifier', 'Classifier Lab'],
    ['retrieval', 'Retrieval Lab'],
    ['generation', 'Generation Lab'],
    ['medical_safety', 'Medical Safety Lab']
  ]
  labs.forEach(([key, title]) => {
    sections.push(buildLabReportMarkdown(result.value, key, title, resultItems.value || []))
    sections.push('\n---\n')
  })
  const runId = result.value?.run?.run_id || `run_${Date.now()}`
  downloadTextFile(`eval_${runId}_full_report.md`, sections.join('\n'))
  ElMessage.success('全链路报告已导出')
}

const exportHistoryReport = () => {
  const payload = historyDetail.value?.result
  if (!payload) {
    ElMessage.warning('请先选择一条历史记录')
    return
  }
  const labKey = activeLabTab.value
  const titles = {
    classifier: 'Classifier Lab',
    retrieval: 'Retrieval Lab',
    generation: 'Generation Lab',
    medical_safety: 'Medical Safety Lab'
  }
  const historyItemsForExport = Array.isArray(historyItems.value) ? historyItems.value : []
  const markdown = buildLabReportMarkdown(payload, labKey, `${titles[labKey] || labKey}（历史）`, historyItemsForExport)
  const historyId = historyDetail.value?.id || `history_${Date.now()}`
  downloadTextFile(`eval_history_${historyId}_${labKey}.md`, markdown)
  ElMessage.success('历史实验室报告已导出')
}

const goBack = () => {
  router.back()
}

const loadLibraries = async () => {
  try {
    const response = await knowledgeAPI.getLibraries()
    if (response.status === 200) {
      libraries.value = response.data || []
    }
  } catch (error) {
    ElMessage.error(error.message || '加载知识库失败')
  }
}

const loadDatasets = async () => {
  try {
    const response = await listEvalDatasets()
    if (response.status === 200) {
      datasets.value = response.data || []
    }
  } catch (error) {
    ElMessage.error(error.message || '加载评测数据集失败')
  }
}

const refreshDatasets = async () => {
  await loadDatasets()
  if (selectedDataset.value !== 'sample' && !datasets.value.find((item) => item.name === selectedDataset.value)) {
    selectedDataset.value = 'sample'
  }
}

const loadHistories = async () => {
  historyLoading.value = true
  try {
    const response = await listEvalHistory({ limit: 50 })
    if (response.status === 200) {
      histories.value = response.data || []
    } else {
      throw new Error(response.msg || '加载评测历史失败')
    }
  } catch (error) {
    ElMessage.error(error.message || '加载评测历史失败')
  } finally {
    historyLoading.value = false
  }
}

const resetHistoryItems = () => {
  historyItems.value = []
  historyItemsPage.value = 1
  historyItemsTotal.value = 0
  historyItemsStats.value = null
}

const loadHistoryItems = async (historyId, page = 1) => {
  if (!historyId) return
  historyItemsLoading.value = true
  try {
    const response = await getEvalHistoryItems(historyId, {
      page,
      page_size: historyItemsPageSize.value
    })
    if (response.status === 200) {
      const payload = response.data || {}
      historyItems.value = payload.items || []
      historyItemsPage.value = payload.page || page
      historyItemsTotal.value = payload.total_items || 0
      historyItemsStats.value = payload.stats || null
    } else {
      throw new Error(response.msg || '加载历史样本失败')
    }
  } catch (error) {
    ElMessage.error(error.message || '加载历史样本失败')
  } finally {
    historyItemsLoading.value = false
  }
}

const changeHistoryItemsPage = async (nextPage) => {
  if (!selectedHistoryId.value) return
  const targetPage = Math.max(1, Math.min(nextPage, historyItemsTotalPages.value))
  if (targetPage === historyItemsPage.value) return
  await loadHistoryItems(selectedHistoryId.value, targetPage)
}

const viewHistory = async (historyId) => {
  if (!historyId) return
  selectedHistoryId.value = historyId
  resetHistoryItems()
  historyDetailLoading.value = true
  try {
    const response = await getEvalHistory(historyId, { include_items: false })
    if (response.status === 200) {
      historyDetail.value = response.data || null
      await loadHistoryItems(historyId, 1)
    } else {
      throw new Error(response.msg || '加载评测历史详情失败')
    }
  } catch (error) {
    ElMessage.error(error.message || '加载评测历史详情失败')
  } finally {
    historyDetailLoading.value = false
  }
}

const applyHistoryResult = async (payload) => {
  let historyPayload = payload?.result || payload
  const historyId = payload?.id
  const needFullItems =
    historyId &&
    (!Array.isArray(historyPayload?.items) || historyPayload.items.length === 0)

  if (needFullItems) {
    try {
      const response = await getEvalHistory(historyId, { include_items: true })
      if (response.status === 200) {
        historyPayload = response.data?.result || historyPayload
      }
    } catch (error) {
      ElMessage.warning(error.message || '加载完整历史详情失败，已使用摘要结果')
    }
  }

  if (historyPayload) {
    result.value = historyPayload
    ElMessage.success('已加载历史评测结果')
  }
}

const clearDataset = () => {
  datasetText.value = ''
}

const saveDataset = async () => {
  const name = datasetName.value.trim()
  if (!name) {
    ElMessage.error('请输入数据集名称')
    return
  }
  if (!datasetText.value.trim()) {
    ElMessage.error('评测数据为空')
    return
  }
  if (datasetExists.value && !allowOverwrite.value) {
    ElMessage.error('同名数据集已存在，请更换名称或允许覆盖')
    return
  }
  saving.value = true
  try {
    const response = await saveEvalDataset({
      name,
      content: datasetText.value,
      overwrite: allowOverwrite.value
    })
    if (response.status === 200) {
      ElMessage.success('评测数据已保存')
      await refreshDatasets()
      if (response.data?.name) {
        selectedDataset.value = response.data.name
      }
    } else {
      throw new Error(response.msg || '保存评测数据失败')
    }
  } catch (error) {
    ElMessage.error(error.message || '保存评测数据失败')
  } finally {
    saving.value = false
  }
}

const loadSelectedDataset = async () => {
  if (selectedDataset.value === 'sample') {
    datasetText.value = sampleDataset
    return
  }
  datasetLoading.value = true
  try {
    const response = await getEvalDatasetContent(selectedDataset.value)
    if (response.status === 200) {
      datasetText.value = response.data?.content || ''
      ElMessage.success('数据集已加载')
    } else {
      throw new Error(response.msg || '加载数据集失败')
    }
  } catch (error) {
    ElMessage.error(error.message || '加载数据集失败')
  } finally {
    datasetLoading.value = false
  }
}

const runClassifierTraining = async ({ autoEvaluate = false } = {}) => {
  if (!datasetText.value.trim()) {
    ElMessage.error('请先准备训练数据（需包含need_retrieval标签）')
    return
  }
  classifierTrainLoading.value = true
  try {
    const response = await trainIntentClassifier({
      dataset_jsonl: datasetText.value,
      eval_dataset_jsonl: classifierTrainUseInlineEval.value ? datasetText.value : null,
      model_name: classifierTrainModelName.value.trim() || null,
      split_strategy: classifierTrainSplitStrategy.value,
      valid_ratio: Number(classifierTrainValidRatio.value),
      test_ratio: Number(classifierTrainTestRatio.value),
      smoothing: Number(classifierTrainSmoothing.value),
      min_token_freq: Number(classifierTrainMinTokenFreq.value)
    })
    if (response.status !== 200) {
      throw new Error(response.msg || '分类器训练失败')
    }
    const payload = response.data || {}
    classifierTrainResult.value = payload
    const recommended = payload.recommended_thresholds || {}
    if (payload.model_path) {
      classifierModelPath.value = payload.model_path
    }
    if (recommended.positive !== undefined && recommended.positive !== null) {
      classifierPositiveThreshold.value = Number(recommended.positive)
    }
    if (recommended.negative !== undefined && recommended.negative !== null) {
      classifierNegativeThreshold.value = Number(recommended.negative)
    }
    ElMessage.success('分类器训练完成，模型与阈值已回填')
    if (autoEvaluate) {
      evalScope.value = 'classifier_only'
      await runEvaluation()
    }
  } catch (error) {
    ElMessage.error(error.message || '分类器训练失败')
  } finally {
    classifierTrainLoading.value = false
  }
}

const runEvaluation = async () => {
  if (!datasetText.value.trim()) {
    ElMessage.error('请输入评测数据')
    return
  }
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
  loading.value = true
  canceling.value = false
  polling.value = false
  result.value = null
  progressEvents.value = []
  taskSubmittedAt.value = null
  taskStartedAt.value = null
  taskRuntimeSeconds.value = null
  taskQueueWaitSeconds.value = null
  activeLabTab.value = defaultLabTabByScope(evalScope.value)
  evaluationProgress.value = {
    stage: 'queued',
    percent: 0,
    processed: 0,
    total: 0,
    message: '任务创建中',
    lab: evalScope.value
  }
  try {
    const response = await startRagEvaluationAsync({
      dataset_jsonl: datasetText.value,
      limit: Number(limit.value) || 0,
      eval_scope: evalScope.value,
      workspace: workspace.value || 'eval_ws',
      retrieval_mode: retrievalMode.value,
      max_retrieval_docs: Number(maxDocs.value) || 3,
      collection_id: collectionId.value || null,
      run_tag: runTag.value.trim() || null,
      enable_ragas: supportsRagas.value ? Boolean(enableRagas.value) : false,
      ragas_limit: supportsRagas.value ? Number(ragasLimit.value) || 0 : 0,
      ci_mode: Boolean(ciMode.value),
      fail_on_gate: Boolean(failOnGate.value || ciMode.value),
      baseline_history_id: baselineHistoryId.value.trim() || null,
      classifier_model_path: classifierModelPath.value.trim() || null,
      classifier_positive_threshold: Number(classifierPositiveThreshold.value),
      classifier_negative_threshold: Number(classifierNegativeThreshold.value),
      include_item_details: Boolean(includeItemDetails.value),
      cache_enabled: Boolean(cacheEnabled.value),
      cache_namespace: cacheNamespace.value.trim() || null,
      dataset_name:
        selectedDataset.value !== 'sample'
          ? selectedDataset.value
          : datasetName.value.trim() || 'sample'
    })
    if (response.status === 200) {
      taskId.value = response.data?.task_id || ''
      if (!taskId.value) {
        throw new Error('未返回任务ID')
      }
      polling.value = true
      const pollOnce = async () => {
        try {
          const statusResponse = await getRagEvaluationStatus(taskId.value)
          if (statusResponse.status !== 200) {
            throw new Error(statusResponse.msg || '获取评测状态失败')
          }
          const statusPayload = statusResponse.data || {}
          taskSubmittedAt.value = statusPayload.submitted_at || taskSubmittedAt.value
          taskStartedAt.value = statusPayload.started_at || taskStartedAt.value
          taskRuntimeSeconds.value = statusPayload.runtime_seconds ?? taskRuntimeSeconds.value
          taskQueueWaitSeconds.value = statusPayload.queue_wait_seconds ?? taskQueueWaitSeconds.value
          if (statusPayload.progress) {
            evaluationProgress.value = {
              ...evaluationProgress.value,
              ...statusPayload.progress
            }
          }
          if (Array.isArray(statusPayload.progress_events)) {
            progressEvents.value = statusPayload.progress_events
          }
          if (statusPayload.status === 'completed') {
            result.value = statusPayload.result
            evaluationProgress.value = {
              ...evaluationProgress.value,
              stage: 'completed',
              percent: 100,
              processed: statusPayload.result?.items_count || statusPayload.result?.total || evaluationProgress.value.processed,
              total: statusPayload.result?.total || statusPayload.result?.items_count || evaluationProgress.value.total,
              message: '评测完成'
            }
            if (statusPayload.result?.warning) {
              ElMessage.warning(statusPayload.result.warning)
            } else {
              ElMessage.success('评测完成')
            }
            await loadHistories()
            polling.value = false
            loading.value = false
            if (pollTimer) {
              clearInterval(pollTimer)
              pollTimer = null
            }
          } else if (statusPayload.status === 'failed') {
            if (statusPayload.result) {
              result.value = statusPayload.result
            }
            evaluationProgress.value = {
              ...evaluationProgress.value,
              stage: 'failed',
              percent: 100,
              message: statusPayload.error || '评测失败'
            }
            polling.value = false
            loading.value = false
            if (pollTimer) {
              clearInterval(pollTimer)
              pollTimer = null
            }
            ElMessage.error(statusPayload.error || '评测失败')
          } else if (statusPayload.status === 'canceled') {
            evaluationProgress.value = {
              ...evaluationProgress.value,
              stage: 'canceled',
              percent: 100,
              message: '评测任务已取消'
            }
            polling.value = false
            loading.value = false
            canceling.value = false
            if (pollTimer) {
              clearInterval(pollTimer)
              pollTimer = null
            }
            ElMessage.warning('评测已取消')
          }
        } catch (error) {
          evaluationProgress.value = {
            ...evaluationProgress.value,
            stage: 'failed',
            percent: 100,
            message: error.message || '获取评测状态失败'
          }
          polling.value = false
          loading.value = false
          canceling.value = false
          if (pollTimer) {
            clearInterval(pollTimer)
            pollTimer = null
          }
          ElMessage.error(error.message || '获取评测状态失败')
        }
      }
      await pollOnce()
      if (polling.value) {
        pollTimer = setInterval(pollOnce, 2000)
      }
    } else {
      throw new Error(response.msg || '评测失败')
    }
  } catch (error) {
    ElMessage.error(error.message || '评测失败')
  } finally {
    if (!polling.value) {
      loading.value = false
    }
  }
}

const cancelEvaluation = async () => {
  if (!taskId.value || !polling.value || canceling.value) {
    return
  }
  canceling.value = true
  try {
    const response = await cancelRagEvaluation(taskId.value)
    if (response.status !== 200) {
      throw new Error(response.msg || '取消评测失败')
    }
    evaluationProgress.value = {
      ...evaluationProgress.value,
      stage: 'canceled',
      percent: 100,
      message: '评测任务已取消'
    }
    polling.value = false
    loading.value = false
    if (pollTimer) {
      clearInterval(pollTimer)
      pollTimer = null
    }
    ElMessage.success('已取消当前评测任务')
  } catch (error) {
    ElMessage.error(error.message || '取消评测失败')
  } finally {
    canceling.value = false
  }
}

onMounted(() => {
  loadLibraries()
  loadDatasets()
  loadHistories()
})

watch(
  () => route.params.lab,
  (value) => {
    const scope = resolveScopeFromRouteLab(value)
    const routeLab = String(value || '').trim().toLowerCase()
    if (!routeLabToScopeMap[routeLab]) {
      syncRouteToScope(scope, true).catch(() => {})
    }
    if (evalScope.value !== scope) {
      evalScope.value = scope
    }
  },
  { immediate: true }
)

watch(
  () => selectedDataset.value,
  (value) => {
    if (value && value !== 'sample' && !datasetName.value.trim()) {
      datasetName.value = value.replace(/\.jsonl$/i, '')
    }
  }
)

watch(
  () => runPreset.value,
  (value) => {
    applyPreset(value)
  },
  { immediate: true }
)

watch(
  () => evalScope.value,
  (value) => {
    if (!supportsRagas.value) {
      enableRagas.value = false
      ragasLimit.value = 0
    }
    progressEvents.value = []
    taskSubmittedAt.value = null
    taskStartedAt.value = null
    taskRuntimeSeconds.value = null
    taskQueueWaitSeconds.value = null
    activeLabTab.value = defaultLabTabByScope(value)
    evaluationProgress.value = {
      stage: 'queued',
      percent: 0,
      processed: 0,
      total: 0,
      message: '',
      lab: value
    }
    applyPreset(runPreset.value)
    syncRouteToScope(value).catch(() => {})
  }
)

watch(
  () => resultModuleLabs.value,
  (value) => {
    const labs = value || {}
    const allowed = resultAllowedLabKeys.value
    if (!allowed.includes(activeLabTab.value) || !labs[activeLabTab.value]) {
      const firstKey = allowed.find((key) => Boolean(labs[key]))
      if (firstKey) {
        activeLabTab.value = firstKey
      }
    }
  }
)

watch(
  () => activeLabTab.value,
  () => {
    labSampleMode.value = 'all'
    labSampleKeyword.value = ''
    labSampleLimit.value = 12
  }
)

onUnmounted(() => {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
})
</script>

<template>
  <div class="min-h-screen bg-slate-50">
    <div class="max-w-7xl mx-auto px-6 py-8 space-y-6">
      <div class="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 class="text-2xl font-semibold text-gray-900">评测实验室</h1>
          <p class="text-sm text-gray-500 mt-2">
            企业级四模块评测：Classifier / Retrieval / Generation / Medical Safety，支持门禁发布与进度追踪
          </p>
        </div>
        <div class="flex items-center gap-3">
          <BaseButton variant="secondary" size="sm" @click="goBack">
            返回
          </BaseButton>
          <div class="px-3 py-1.5 rounded-full bg-white border border-gray-200 text-xs text-gray-600">
            数据条数 {{ datasetCount }}
          </div>
          <BaseButton
            v-if="polling"
            :loading="canceling"
            variant="secondary"
            @click="cancelEvaluation"
          >
            {{ canceling ? '取消中...' : '取消评测' }}
          </BaseButton>
          <BaseButton :loading="loading" variant="primary" @click="runEvaluation">
            {{ loading ? '评测中...' : '运行评测' }}
          </BaseButton>
        </div>
      </div>

      <div class="rounded-xl border border-gray-200 bg-white p-3">
        <div class="flex items-center justify-between">
          <div class="text-sm font-semibold text-gray-900">企业评测模块</div>
          <div class="text-xs text-gray-500">点击切换执行范围</div>
        </div>
        <div class="mt-3 grid grid-cols-1 md:grid-cols-5 gap-2">
          <button
            v-for="option in evalScopeOptions"
            :key="`scope-top-tab-${option.value}`"
            class="rounded-lg border px-2.5 py-2 text-left transition-colors"
            :class="evalScope === option.value ? 'border-slate-900 bg-slate-900 text-white' : 'border-gray-200 bg-white text-gray-800 hover:border-slate-400'"
            @click="selectScope(option.value)"
          >
            <div class="text-xs font-semibold">{{ option.title }}</div>
            <div class="text-[11px] mt-1 opacity-80">{{ option.subtitle }}</div>
          </button>
        </div>
      </div>

      <div
        v-if="loading || polling || progressPercent > 0"
        class="rounded-xl border border-gray-200 bg-white px-4 py-3 space-y-3"
      >
        <div class="flex items-center justify-between text-xs text-gray-600">
          <div>
            进度：{{ formatProgressStage(evaluationProgress?.stage) }}
            <span v-if="evaluationProgress?.message"> · {{ evaluationProgress?.message }}</span>
            <span v-if="evaluationProgress?.lab"> · {{ evaluationProgress?.lab }}</span>
          </div>
          <div>
            {{ formatNumber(progressPercent, 1) }}%
            <span v-if="evaluationProgress?.total">（{{ evaluationProgress?.processed || 0 }}/{{ evaluationProgress?.total }}）</span>
          </div>
        </div>
        <div class="h-2 w-full rounded-full bg-gray-100 overflow-hidden">
          <div
            class="h-full bg-slate-900 transition-all duration-300"
            :style="{ width: `${progressPercent}%` }"
          ></div>
        </div>
        <div class="grid grid-cols-2 gap-2 md:grid-cols-7">
          <div
            v-for="step in progressTimeline"
            :key="`progress-step-${step.key}`"
            class="rounded-lg border px-2 py-1 text-[11px] text-center transition-colors"
            :class="progressStepClass(step.status)"
          >
            {{ step.label }}
          </div>
        </div>
        <div class="grid grid-cols-1 gap-3 lg:grid-cols-[1fr_1.2fr]">
          <div class="rounded-lg border border-gray-200 bg-gray-50 p-3 space-y-2 text-xs text-gray-600">
            <div>当前阶段耗时：{{ formatDurationSec(currentStageElapsedSec) }}</div>
            <div>预计剩余：{{ formatDurationSec(progressEtaSec) }}</div>
            <div>提交时间：{{ formatProgressTime(taskSubmittedAt) }}</div>
            <div>开始时间：{{ formatProgressTime(taskStartedAt) }}</div>
            <div>排队耗时：{{ formatDurationSec(taskQueueWaitSeconds) }}</div>
            <div>任务总耗时：{{ formatDurationSec(taskRuntimeSeconds) }}</div>
          </div>
          <div class="rounded-lg border border-gray-200 bg-white p-3">
            <div class="text-xs font-semibold text-gray-700">实时执行日志</div>
            <div class="mt-2 space-y-2 max-h-40 overflow-auto pr-1">
              <div
                v-for="(event, idx) in progressEventRows"
                :key="`progress-event-${idx}`"
                class="rounded-lg border px-2 py-1.5 text-[11px]"
                :class="progressBadgeClass(event.stage)"
              >
                <div class="flex items-center justify-between gap-2">
                  <span>{{ formatProgressStage(event.stage) }}</span>
                  <span>{{ formatProgressTime(event.ts) }}</span>
                </div>
                <div class="mt-1">
                  {{ event.message || '--' }}
                  <span v-if="event.total">（{{ event.processed || 0 }}/{{ event.total }}）</span>
                </div>
              </div>
              <div v-if="progressEventRows.length === 0" class="text-[11px] text-gray-500">
                暂无日志
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <BaseCard class="space-y-6">
          <div class="flex items-center justify-between">
            <h2 class="text-lg font-semibold text-gray-900">评测配置</h2>
            <span class="text-xs text-gray-500">支持JSONL格式</span>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label class="text-xs font-medium text-gray-500">运行预设</label>
              <select v-model="runPreset" class="input mt-2">
                <option value="smoke">Smoke（快速验证）</option>
                <option value="standard">Standard（默认）</option>
                <option value="full">Full（全量回归）</option>
              </select>
            </div>
            <div class="rounded-lg border border-gray-200 bg-gray-50 px-3 py-2.5 text-xs text-gray-600">
              当前模块：{{ selectedScopeOption?.title || '--' }}
            </div>
            <div v-if="!isClassifierOnly">
              <label class="text-xs font-medium text-gray-500">知识库</label>
              <select v-model="collectionId" class="input mt-2" :disabled="isClassifierOnly">
                <option value="">默认知识库</option>
                <option v-for="library in libraries" :key="library.id" :value="library.collection_id">
                  {{ library.title }}
                </option>
              </select>
              <div v-if="isClassifierOnly" class="text-[11px] text-gray-400 mt-2">
                小模型验证模式下不会调用知识库检索。
              </div>
            </div>
            <div v-if="!isClassifierOnly">
              <label class="text-xs font-medium text-gray-500">检索模式</label>
              <select v-model="retrievalMode" class="input mt-2" :disabled="isClassifierOnly">
                <option value="vector_only">向量检索</option>
                <option value="hybrid">融合检索</option>
                <option value="graph_only">图检索</option>
                <option value="auto">自动选择</option>
                <option value="no_retrieval">不检索</option>
              </select>
            </div>
            <div v-if="!isClassifierOnly">
              <label class="text-xs font-medium text-gray-500">最大检索文档</label>
              <input v-model="maxDocs" type="number" min="1" class="input mt-2" :disabled="isClassifierOnly" />
            </div>
            <div>
              <label class="text-xs font-medium text-gray-500">评测条数上限</label>
              <input v-model="limit" type="number" min="0" class="input mt-2" />
            </div>
            <div v-if="!isClassifierOnly" class="md:col-span-2">
              <label class="text-xs font-medium text-gray-500">评测工作区</label>
              <select v-model="workspace" class="input mt-2" :disabled="isClassifierOnly">
                <option v-for="option in workspaceOptions" :key="option.value" :value="option.value">
                  {{ option.label }}
                </option>
              </select>
              <div class="text-[11px] text-gray-400 mt-2">
                {{ isClassifierOnly ? '小模型验证模式下该配置不会生效。' : '用于隔离评测运行的存储空间，与评测数据集无关' }}
              </div>
            </div>
            <div>
              <label class="text-xs font-medium text-gray-500">运行标签</label>
              <input v-model="runTag" type="text" class="input mt-2" placeholder="例如：med-v2-baseline" />
            </div>
            <div>
              <label class="text-xs font-medium text-gray-500">基线对比ID（可选）</label>
              <input
                v-model="baselineHistoryId"
                type="text"
                class="input mt-2"
                placeholder="留空则自动选择最近基线"
              />
            </div>
            <div v-if="supportsRagas">
              <label class="text-xs font-medium text-gray-500">RAGAS采样条数</label>
              <input
                v-model="ragasLimit"
                type="number"
                min="0"
                class="input mt-2"
                placeholder="0表示全量评测"
                :disabled="isClassifierOnly"
              />
              <div class="text-[11px] text-gray-400 mt-2">
                0 为全量；默认 10 条可显著缩短评测时长
              </div>
            </div>
            <div>
              <label class="text-xs font-medium text-gray-500">缓存命名空间</label>
              <input v-model="cacheNamespace" type="text" class="input mt-2" placeholder="例如：eval-default" />
            </div>
            <div>
              <label class="text-xs font-medium text-gray-500">分类器模型路径（可选）</label>
              <input
                v-model="classifierModelPath"
                type="text"
                class="input mt-2"
                placeholder="留空则使用后端默认模型"
              />
            </div>
            <div>
              <label class="text-xs font-medium text-gray-500">分类器正阈值</label>
              <input v-model="classifierPositiveThreshold" type="number" min="0" max="1" step="0.01" class="input mt-2" />
            </div>
            <div>
              <label class="text-xs font-medium text-gray-500">分类器负阈值</label>
              <input v-model="classifierNegativeThreshold" type="number" min="0" max="1" step="0.01" class="input mt-2" />
            </div>
            <div class="md:col-span-2 grid grid-cols-1 md:grid-cols-5 gap-3">
              <label v-if="supportsRagas" class="flex items-center gap-2 text-xs text-gray-600">
                <input v-model="enableRagas" type="checkbox" class="h-4 w-4" :disabled="isClassifierOnly" />
                <span>启用 RAGAS 指标</span>
              </label>
              <label class="flex items-center gap-2 text-xs text-gray-600">
                <input v-model="includeItemDetails" type="checkbox" class="h-4 w-4" />
                <span>返回样本详情</span>
              </label>
              <label class="flex items-center gap-2 text-xs text-gray-600">
                <input v-model="cacheEnabled" type="checkbox" class="h-4 w-4" />
                <span>启用缓存标记</span>
              </label>
              <label class="flex items-center gap-2 text-xs text-gray-600">
                <input v-model="ciMode" type="checkbox" class="h-4 w-4" />
                <span>CI模式（门禁默认阻断）</span>
              </label>
              <label class="flex items-center gap-2 text-xs text-gray-600">
                <input v-model="failOnGate" type="checkbox" class="h-4 w-4" />
                <span>门禁失败阻断</span>
              </label>
            </div>
          </div>
          <div
            v-if="isClassifierOnly"
            class="rounded-lg border border-cyan-200 bg-cyan-50 px-3 py-2 text-xs text-cyan-800"
          >
            当前为小模型验证模式：将只执行 Classifier Lab，不会调用检索与生成链路。
          </div>
          <div
            v-else-if="evalScope !== 'rag_full'"
            class="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800"
          >
            当前为单实验室模式：仅执行 {{ evalScope }}，用于快速定位模块问题；全链路回归请切换到 RAG全链路验证。
          </div>

          <div class="rounded-xl border border-gray-200 bg-white p-4 space-y-3">
            <div class="flex flex-wrap items-center justify-between gap-2">
              <div>
                <div class="text-sm font-semibold text-gray-900">意图分类训练器</div>
                <div class="text-xs text-gray-500 mt-1">
                  在评测页完成训练 -> 阈值建议 -> Classifier Lab 评测
                </div>
              </div>
              <div class="flex items-center gap-2">
                <BaseButton
                  size="sm"
                  variant="secondary"
                  :loading="classifierTrainLoading"
                  @click="runClassifierTraining({ autoEvaluate: false })"
                >
                  {{ classifierTrainLoading ? '训练中...' : '训练模型' }}
                </BaseButton>
                <BaseButton
                  size="sm"
                  variant="primary"
                  :loading="classifierTrainLoading"
                  @click="runClassifierTraining({ autoEvaluate: true })"
                >
                  训练并评测
                </BaseButton>
              </div>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
              <div>
                <label class="text-xs font-medium text-gray-500">模型名（可选）</label>
                <input
                  v-model="classifierTrainModelName"
                  type="text"
                  class="input mt-2"
                  placeholder="例如：retrieval_intent_med_v2"
                />
              </div>
              <div>
                <label class="text-xs font-medium text-gray-500">切分策略</label>
                <select v-model="classifierTrainSplitStrategy" class="input mt-2">
                  <option value="random">random</option>
                  <option value="time">time</option>
                </select>
              </div>
              <div>
                <label class="text-xs font-medium text-gray-500">valid_ratio</label>
                <input v-model="classifierTrainValidRatio" type="number" min="0" max="0.9" step="0.05" class="input mt-2" />
              </div>
              <div>
                <label class="text-xs font-medium text-gray-500">test_ratio</label>
                <input v-model="classifierTrainTestRatio" type="number" min="0" max="0.9" step="0.05" class="input mt-2" />
              </div>
              <div>
                <label class="text-xs font-medium text-gray-500">smoothing</label>
                <input v-model="classifierTrainSmoothing" type="number" min="0.1" step="0.1" class="input mt-2" />
              </div>
              <div>
                <label class="text-xs font-medium text-gray-500">min_token_freq</label>
                <input v-model="classifierTrainMinTokenFreq" type="number" min="1" step="1" class="input mt-2" />
              </div>
              <label class="flex items-center gap-2 text-xs text-gray-600 md:col-span-2">
                <input v-model="classifierTrainUseInlineEval" type="checkbox" class="h-4 w-4" />
                <span>训练后使用当前数据做内联分类评测（快速验证）</span>
              </label>
            </div>
            <div v-if="classifierTrainResult" class="rounded-lg border border-gray-200 bg-gray-50 p-3 space-y-2 text-xs">
              <div class="text-gray-700 break-all">
                模型路径：{{ classifierTrainResult?.model_path || '--' }}
              </div>
              <div class="grid grid-cols-2 md:grid-cols-4 gap-2 text-gray-700">
                <div>建议正阈值 {{ formatNumber(classifierTrainResult?.recommended_thresholds?.positive, 3) }}</div>
                <div>建议负阈值 {{ formatNumber(classifierTrainResult?.recommended_thresholds?.negative, 3) }}</div>
                <div>训练集 {{ classifierTrainResult?.split?.train_size ?? '--' }}</div>
                <div>验证集 {{ classifierTrainResult?.split?.valid_size ?? '--' }}</div>
              </div>
              <div class="grid grid-cols-1 md:grid-cols-3 gap-2 text-gray-700">
                <div class="rounded border border-gray-200 bg-white px-2 py-1.5">
                  train F1 {{ formatNumber(classifierTrainMetrics?.train?.f1, 3) }} · PR-AUC {{ formatNumber(classifierTrainMetrics?.train?.pr_auc, 3) }}
                </div>
                <div class="rounded border border-gray-200 bg-white px-2 py-1.5">
                  valid F1 {{ formatNumber(classifierTrainMetrics?.valid?.f1, 3) }} · PR-AUC {{ formatNumber(classifierTrainMetrics?.valid?.pr_auc, 3) }}
                </div>
                <div class="rounded border border-gray-200 bg-white px-2 py-1.5">
                  test F1 {{ formatNumber(classifierTrainMetrics?.test?.f1, 3) }} · PR-AUC {{ formatNumber(classifierTrainMetrics?.test?.pr_auc, 3) }}
                </div>
              </div>
            </div>
          </div>

          <div>
            <div class="flex items-center justify-between">
              <h3 class="text-sm font-semibold text-gray-900">评测数据</h3>
              <div class="flex gap-2">
                <button class="text-xs text-gray-500 hover:text-gray-700" @click="refreshDatasets">
                  刷新列表
                </button>
                <button class="text-xs text-gray-500 hover:text-gray-700" @click="clearDataset">
                  清空
                </button>
              </div>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-[1.4fr_1fr] gap-3 mt-3">
              <div>
                <label class="text-xs font-medium text-gray-500">评测数据集</label>
                <select v-model="selectedDataset" class="input mt-2">
                  <option value="sample">内置示例</option>
                  <option v-for="dataset in datasets" :key="dataset.name" :value="dataset.name">
                    {{ dataset.name }} · {{ dataset.lines || 0 }}条 · {{ dataset.source === 'builtin' ? '内置' : '我的' }}
                  </option>
                </select>
              </div>
              <div class="flex items-end gap-2">
                <BaseButton
                  size="sm"
                  variant="secondary"
                  :loading="datasetLoading"
                  @click="loadSelectedDataset"
                >
                  {{ datasetLoading ? '加载中' : '加载数据集' }}
                </BaseButton>
                <div class="text-xs text-gray-400">
                  <span v-if="selectedDatasetInfo">
                    {{ selectedDatasetInfo.lines || 0 }} 条
                    <span v-if="selectedDatasetInfo.size !== null">
                      · {{ formatBytes(selectedDatasetInfo.size) }}
                    </span>
                    <span v-if="selectedDatasetInfo.source === 'builtin'">
                      · 内置只读
                    </span>
                  </span>
                </div>
              </div>
            </div>
            <textarea
              v-model="datasetText"
              rows="10"
              class="input mt-3 font-mono text-xs"
              placeholder='{"question":"...","reference":"..."}'
            ></textarea>
            <div class="mt-4 grid grid-cols-1 md:grid-cols-[1fr_auto] gap-3 items-end">
              <div>
                <label class="text-xs font-medium text-gray-500">保存为评测数据集</label>
                <input
                  v-model="datasetName"
                  type="text"
                  class="input mt-2"
                  placeholder="例如：medic_eval"
                />
                <div class="flex items-center gap-2 mt-2 text-xs text-gray-500">
                  <input v-model="allowOverwrite" type="checkbox" class="h-4 w-4" />
                  <span>允许覆盖同名数据集</span>
                  <span v-if="datasetExists" class="text-amber-600">同名已存在</span>
                </div>
              </div>
              <BaseButton size="sm" variant="secondary" :loading="saving" @click="saveDataset">
                保存数据集
              </BaseButton>
            </div>
          </div>
        </BaseCard>

        <BaseCard class="space-y-6">
          <div class="flex items-center justify-between">
            <h2 class="text-lg font-semibold text-gray-900">评测结果</h2>
            <div class="flex items-center gap-2">
              <div class="text-xs text-gray-500" v-if="result">
                共 {{ resultTotal }} 条，用时 {{ resultElapsedMs }} ms
              </div>
              <BaseButton
                size="sm"
                variant="secondary"
                :disabled="!result"
                @click="exportActiveLabReport"
              >
                导出当前实验室
              </BaseButton>
              <BaseButton
                size="sm"
                variant="secondary"
                :disabled="!result || resultIsClassifierOnly"
                @click="exportFullRunReport"
              >
                导出全链路报告
              </BaseButton>
            </div>
          </div>

          <div v-if="!resultIsClassifierOnly" class="grid grid-cols-2 gap-4">
            <div
              v-for="metric in metricDefinitions"
              :key="metric.key"
              class="rounded-xl border border-gray-100 bg-white p-4"
            >
              <div class="flex items-center justify-between text-xs text-gray-500">
                <span>{{ metric.label }}</span>
                <span class="px-2 py-0.5 rounded-full text-xs"
                  :class="`bg-${metric.tone}-50 text-${metric.tone}-600`"
                >
                  {{ metric.key }}
                </span>
              </div>
              <div class="mt-3 text-2xl font-semibold text-gray-900">
                {{ formatScore(metricValue(metric.key)) }}
              </div>
            </div>
          </div>

          <div v-if="moduleLabEntries.length > 0" class="rounded-xl border border-gray-100 bg-white p-3 space-y-3">
            <div class="flex flex-wrap items-center justify-between gap-2">
              <div class="text-sm font-semibold text-gray-900">{{ resultIsClassifierOnly ? 'Classifier Lab' : '四大实验室' }}</div>
              <div class="text-xs text-gray-500">
                {{ resultIsClassifierOnly ? '当前仅展示分类器评测链路' : '按模块查看指标、门禁与诊断' }}
              </div>
            </div>
            <div v-if="moduleLabEntries.length > 1" class="flex flex-wrap gap-2">
              <button
                v-for="lab in moduleLabEntries"
                :key="`lab-tab-${lab.key}`"
                class="px-2.5 py-1 rounded-full border text-xs"
                :class="activeLabTab === lab.key ? 'bg-slate-900 text-white border-slate-900' : 'bg-white text-slate-700 border-slate-200'"
                @click="activeLabTab = lab.key"
              >
                {{ lab.label }}
              </button>
            </div>
            <div v-if="activeModuleLab" class="grid grid-cols-1 md:grid-cols-[1fr_auto] gap-3 text-xs">
              <div class="rounded-lg border border-gray-200 bg-gray-50 p-3 space-y-2">
                <div class="text-gray-700 font-medium">{{ activeModuleLab?.name || '--' }}</div>
                <div class="flex flex-wrap gap-2">
                  <span class="px-2 py-0.5 rounded-full border" :class="labEnabledClass(activeModuleLab)">
                    {{ labEnabledLabel(activeModuleLab) }}
                  </span>
                  <span
                    v-if="activeModuleLab?.quality_gate?.overall_status"
                    class="px-2 py-0.5 rounded-full border"
                    :class="gateStatusClass(activeModuleLab?.quality_gate?.overall_status)"
                  >
                    门禁 {{ gateStatusLabel(activeModuleLab?.quality_gate?.overall_status) }}
                  </span>
                  <span
                    v-if="activeModuleLab?.hard_gate?.overall_status"
                    class="px-2 py-0.5 rounded-full border"
                    :class="gateStatusClass(activeModuleLab?.hard_gate?.overall_status)"
                  >
                    安全硬门禁 {{ gateStatusLabel(activeModuleLab?.hard_gate?.overall_status) }}
                  </span>
                </div>
                <div class="text-gray-500 break-all">{{ activeModuleLab?.reason || '--' }}</div>
              </div>
              <div class="rounded-lg border border-gray-200 bg-white p-3 text-xs">
                <pre class="whitespace-pre-wrap text-[11px] text-gray-700">{{ JSON.stringify(activeModuleLab?.metrics || activeModuleLab?.summary || {}, null, 2) }}</pre>
              </div>
            </div>
            <div v-if="activeModuleLab" class="rounded-lg border border-gray-200 bg-gray-50 p-3 space-y-3 text-xs">
              <div class="text-gray-700 font-medium">实验室工作台</div>
              <div class="grid grid-cols-2 md:grid-cols-4 gap-2">
                <div
                  v-for="metric in activeLabQuickMetrics"
                  :key="`active-lab-metric-${metric.key}`"
                  class="rounded-lg border border-gray-200 bg-white px-2 py-1.5"
                >
                  <div class="text-gray-500">{{ metric.label }}</div>
                  <div class="mt-1 font-semibold text-gray-900">{{ metric.value ?? '--' }}</div>
                </div>
              </div>
              <div class="grid grid-cols-1 md:grid-cols-[200px_1fr_120px] gap-2">
                <select v-model="labSampleMode" class="input">
                  <option
                    v-for="mode in activeLabSampleModes"
                    :key="`lab-mode-${mode.value}`"
                    :value="mode.value"
                  >
                    {{ mode.label }}
                  </option>
                </select>
                <input
                  v-model="labSampleKeyword"
                  type="text"
                  class="input"
                  placeholder="按问题/回答/badcase标签过滤"
                />
                <input
                  v-model="labSampleLimit"
                  type="number"
                  min="1"
                  max="50"
                  class="input"
                  placeholder="样本数"
                />
              </div>
              <div v-if="activeLabSampleItems.length === 0" class="text-gray-500">
                当前筛选条件下无样本
              </div>
              <div v-else class="space-y-2 max-h-56 overflow-auto pr-1">
                <div
                  v-for="(item, idx) in activeLabSampleItems"
                  :key="`active-lab-item-${idx}`"
                  class="rounded-lg border border-gray-200 bg-white px-2 py-2"
                >
                  <div class="text-gray-800">{{ item.question }}</div>
                  <div class="mt-1 text-[11px] text-gray-500">
                    路由 {{ formatRoutingStageLabel(item.routing?.stage) }} · 上下文 {{ item.contexts_count || 0 }} ·
                    延迟 {{ item.latency_ms ?? '--' }} ms
                  </div>
                  <div class="mt-1 flex flex-wrap gap-1">
                    <span
                      v-for="tag in item.badcase?.tags || []"
                      :key="`active-lab-item-tag-${idx}-${tag}`"
                      class="px-1.5 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200"
                    >
                      {{ tag }}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div
            v-if="resultIsClassifierOnly && (resultClassifier || resultClassifierGate || resultRouting || resultStability)"
            class="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs"
          >
            <div v-if="resultClassifier" class="rounded-xl border border-gray-100 bg-slate-50 p-3 space-y-2">
              <div class="text-gray-500">分类器核心指标</div>
              <div class="grid grid-cols-2 gap-2 text-gray-700">
                <div>标注样本 {{ resultClassifier?.labeled_items ?? '--' }}</div>
                <div>准确率 {{ formatPercent(resultClassifier?.accuracy) }}</div>
                <div>精确率 {{ formatPercent(resultClassifier?.precision) }}</div>
                <div>召回率 {{ formatPercent(resultClassifier?.recall) }}</div>
                <div>F1 {{ formatNumber(resultClassifier?.f1, 3) }}</div>
                <div>不确定率 {{ formatPercent(resultClassifier?.uncertain_rate) }}</div>
              </div>
            </div>
            <div v-if="resultRouting" class="rounded-xl border border-gray-100 bg-slate-50 p-3 space-y-2">
              <div class="text-gray-500">路由稳定性（分类器）</div>
              <div class="grid grid-cols-2 gap-2 text-gray-700">
                <div>统计分类器决策率 {{ formatPercent(resultRouting?.statistical_classifier_decision_rate) }}</div>
                <div>LLM兜底率 {{ formatPercent(resultRouting?.llm_fallback_rate) }}</div>
                <div>统计均值概率 {{ formatNumber(resultRouting?.avg_statistical_probability, 3) }}</div>
                <div>需检索占比 {{ formatPercent(resultRouting?.need_retrieval_rate) }}</div>
              </div>
            </div>
            <div v-if="resultStability" class="rounded-xl border border-gray-100 bg-slate-50 p-3 space-y-2">
              <div class="text-gray-500">执行稳定性</div>
              <div class="grid grid-cols-2 gap-2 text-gray-700">
                <div>失败数 {{ resultStability?.failed_items ?? '--' }}</div>
                <div>错误率 {{ formatPercent(resultStability?.error_rate) }}</div>
                <div>空回答 {{ resultStability?.empty_answer_count ?? '--' }}</div>
                <div>空回答率 {{ formatPercent(resultStability?.empty_answer_rate) }}</div>
              </div>
            </div>
            <div v-if="resultClassifierGate" class="rounded-xl border border-gray-100 bg-slate-50 p-3 space-y-2">
              <div class="text-gray-500">分类器门禁</div>
              <div class="text-[11px] text-gray-500">
                门禁 {{ gateStatusLabel(resultClassifierGate?.overall_status) }} · 通过 {{ resultClassifierGate?.pass_count ?? 0 }} · 失败 {{ resultClassifierGate?.fail_count ?? 0 }}
              </div>
            </div>
          </div>

          <div v-if="!resultIsClassifierOnly && (resultPerformance || resultRetrieval || resultRouting || resultStability || resultSop || resultAnswerOverlap || resultCost || resultCache || resultClassifier || resultEvalLabs)" class="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
            <div class="rounded-xl border border-gray-100 bg-slate-50 p-3 space-y-2">
              <div class="text-gray-500">性能</div>
              <div class="grid grid-cols-2 gap-2 text-gray-700">
                <div>平均延迟 {{ formatNumber(resultPerformance?.avg_latency_ms) }} ms</div>
                <div>P95 {{ formatNumber(resultPerformance?.p95_latency_ms) }} ms</div>
                <div>P99 {{ formatNumber(resultPerformance?.p99_latency_ms) }} ms</div>
                <div>吞吐 {{ formatNumber(resultPerformance?.throughput_qps, 3) }} qps</div>
              </div>
            </div>
            <div class="rounded-xl border border-gray-100 bg-slate-50 p-3 space-y-2">
              <div class="text-gray-500">检索</div>
              <div class="grid grid-cols-2 gap-2 text-gray-700">
                <div>平均上下文 {{ formatNumber(resultRetrieval?.avg_contexts_per_item) }}</div>
                <div>上下文覆盖 {{ formatPercent(resultRetrieval?.context_presence_rate) }}</div>
                <div>P95上下文 {{ formatNumber(resultRetrieval?.p95_contexts_per_item) }}</div>
                <div>平均上下文字数 {{ formatNumber(resultRetrieval?.avg_context_chars_per_item) }}</div>
                <div>标注上下文命中 {{ formatPercent(resultRetrieval?.labeled_context_hit_rate) }}</div>
                <div>标注来源命中 {{ formatPercent(resultRetrieval?.labeled_source_hit_rate) }}</div>
              </div>
              <div class="flex flex-wrap gap-2 text-[11px] text-gray-600">
                <span v-for="[sourceName, count] in sourceEntries" :key="sourceName" class="px-2 py-0.5 rounded-full bg-white border border-gray-200">
                  {{ sourceName }} {{ count }}
                </span>
              </div>
            </div>
            <div class="rounded-xl border border-gray-100 bg-slate-50 p-3 space-y-2">
              <div class="text-gray-500">路由决策</div>
              <div class="grid grid-cols-2 gap-2 text-gray-700">
                <div>需检索占比 {{ formatPercent(resultRouting?.need_retrieval_rate) }}</div>
                <div>LLM兜底率 {{ formatPercent(resultRouting?.llm_fallback_rate) }}</div>
                <div>统计分类器决策率 {{ formatPercent(resultRouting?.statistical_classifier_decision_rate) }}</div>
                <div>统计均值概率 {{ formatNumber(resultRouting?.avg_statistical_probability, 3) }}</div>
              </div>
              <div class="flex flex-wrap gap-2 text-[11px]">
                <span
                  v-for="(count, stage) in resultRouting?.stage_distribution || {}"
                  :key="`routing-stage-${stage}`"
                  class="px-2 py-0.5 rounded-full border"
                  :class="routingStageClass(stage)"
                >
                  {{ formatRoutingStageLabel(stage) }} {{ count }}
                </span>
              </div>
            </div>
            <div class="rounded-xl border border-gray-100 bg-slate-50 p-3 space-y-2">
              <div class="text-gray-500">稳定性</div>
              <div class="grid grid-cols-2 gap-2 text-gray-700">
                <div>失败数 {{ resultStability?.failed_items ?? '--' }}</div>
                <div>错误率 {{ formatPercent(resultStability?.error_rate) }}</div>
                <div>空回答 {{ resultStability?.empty_answer_count ?? '--' }}</div>
                <div>空回答率 {{ formatPercent(resultStability?.empty_answer_rate) }}</div>
              </div>
            </div>
            <div class="rounded-xl border border-gray-100 bg-slate-50 p-3 space-y-2">
              <div class="text-gray-500">医疗SOP安全</div>
              <div class="grid grid-cols-2 gap-2 text-gray-700">
                <div>医疗问题占比 {{ formatPercent(resultSop?.medical_item_rate) }}</div>
                <div>转人工率 {{ formatPercent(resultSop?.handoff_rate) }}</div>
                <div>结构化有效率 {{ formatPercent(resultSop?.structured_decision_valid_rate) }}</div>
                <div>平均红线数 {{ formatNumber(resultSop?.avg_red_flags_per_item, 3) }}</div>
              </div>
              <div class="text-[11px] text-gray-500">
                标注转人工准确率 {{ formatPercent(resultSop?.handoff_accuracy) }}
              </div>
            </div>
            <div class="rounded-xl border border-gray-100 bg-slate-50 p-3 space-y-2">
              <div class="text-gray-500">答案重合（轻量）</div>
              <div class="grid grid-cols-2 gap-2 text-gray-700">
                <div>覆盖率 {{ formatPercent(resultAnswerOverlap?.answer_overlap_coverage_rate) }}</div>
                <div>平均F1 {{ formatNumber(resultAnswerOverlap?.avg_answer_overlap_f1, 3) }}</div>
                <div>平均Precision {{ formatNumber(resultAnswerOverlap?.avg_answer_overlap_precision, 3) }}</div>
                <div>平均Recall {{ formatNumber(resultAnswerOverlap?.avg_answer_overlap_recall, 3) }}</div>
              </div>
            </div>
            <div class="rounded-xl border border-gray-100 bg-slate-50 p-3 space-y-2">
              <div class="text-gray-500">成本与缓存</div>
              <div class="grid grid-cols-2 gap-2 text-gray-700">
                <div>上下文总字数 {{ resultCost?.estimated?.total_context_chars ?? '--' }}</div>
                <div>回答总字数 {{ resultCost?.estimated?.total_answer_chars ?? '--' }}</div>
                <div>平均回答字数 {{ formatNumber(resultCost?.estimated?.avg_answer_chars_per_item) }}</div>
                <div>缓存开关 {{ resultCache?.enabled ? '开启' : '关闭' }}</div>
              </div>
              <div class="text-[11px] text-gray-500">
                命名空间 {{ resultCache?.namespace || '--' }}
              </div>
            </div>
            <div v-if="resultClassifier" class="rounded-xl border border-gray-100 bg-slate-50 p-3 space-y-2">
              <div class="text-gray-500">小模型验证</div>
              <div class="grid grid-cols-2 gap-2 text-gray-700">
                <div>标注样本 {{ resultClassifier?.labeled_items ?? '--' }}</div>
                <div>准确率 {{ formatPercent(resultClassifier?.accuracy) }}</div>
                <div>精确率 {{ formatPercent(resultClassifier?.precision) }}</div>
                <div>召回率 {{ formatPercent(resultClassifier?.recall) }}</div>
                <div>F1 {{ formatNumber(resultClassifier?.f1, 3) }}</div>
                <div>不确定率 {{ formatPercent(resultClassifier?.uncertain_rate) }}</div>
              </div>
              <div v-if="resultClassifierGate" class="text-[11px] text-gray-500">
                门禁 {{ gateStatusLabel(resultClassifierGate?.overall_status) }} · 通过 {{ resultClassifierGate?.pass_count ?? 0 }} · 失败 {{ resultClassifierGate?.fail_count ?? 0 }}
              </div>
            </div>
            <div v-if="resultEvalLabs" class="rounded-xl border border-gray-100 bg-slate-50 p-3 space-y-2">
              <div class="text-gray-500">实验室模块</div>
              <div class="flex flex-wrap gap-2 text-[11px]">
                <span class="px-2 py-0.5 rounded-full bg-white border border-gray-200 text-gray-700">
                  分类器验证 {{ resultEvalLabs?.classifier_validation?.enabled ? '已启用' : '未启用' }}
                </span>
                <span class="px-2 py-0.5 rounded-full bg-white border border-gray-200 text-gray-700">
                  检索验证
                </span>
                <span class="px-2 py-0.5 rounded-full bg-white border border-gray-200 text-gray-700">
                  生成验证
                </span>
                <span class="px-2 py-0.5 rounded-full bg-white border border-gray-200 text-gray-700">
                  医疗安全验证
                </span>
              </div>
            </div>
            <div v-if="resultBadcase" class="rounded-xl border border-gray-100 bg-slate-50 p-3 space-y-2">
              <div class="text-gray-500">Badcase 概览</div>
              <div class="grid grid-cols-2 gap-2 text-gray-700">
                <div>数量 {{ resultBadcase?.badcase_count ?? '--' }}</div>
                <div>占比 {{ formatPercent(resultBadcase?.badcase_rate) }}</div>
                <div>高优先级 {{ resultBadcase?.severity_distribution?.high ?? 0 }}</div>
                <div>中优先级 {{ resultBadcase?.severity_distribution?.medium ?? 0 }}</div>
              </div>
              <div class="flex flex-wrap gap-2 text-[11px]">
                <span
                  v-for="tagItem in resultBadcase?.top_tags || []"
                  :key="`badcase-tag-${tagItem.tag}`"
                  class="px-2 py-0.5 rounded-full bg-white border border-gray-200 text-gray-700"
                >
                  {{ tagItem.tag }} {{ tagItem.count }}
                </span>
              </div>
            </div>
            <div v-if="resultReproducibility" class="rounded-xl border border-gray-100 bg-slate-50 p-3 space-y-2">
              <div class="text-gray-500">可复现快照</div>
              <div class="grid grid-cols-1 gap-1 text-[11px] text-gray-700">
                <div>Replay Key {{ resultReproducibility?.replay_key || '--' }}</div>
                <div>Dataset Hash {{ formatHashShort(resultReproducibility?.dataset_sha256) }}</div>
                <div>Config Hash {{ formatHashShort(resultReproducibility?.config_sha256) }}</div>
                <div>Collection {{ resultReproducibility?.collection_snapshot_id || '--' }}</div>
              </div>
            </div>
          </div>

          <div v-if="!resultIsClassifierOnly && resultQualityGates" class="rounded-xl border border-gray-100 bg-white p-3 space-y-3 text-xs">
            <div class="flex items-center justify-between">
              <div class="text-gray-700 font-semibold">质量门禁</div>
              <span
                class="px-2 py-0.5 rounded-full border"
                :class="gateStatusClass(resultQualityGates?.overall_status)"
              >
                {{ gateStatusLabel(resultQualityGates?.overall_status) }}
              </span>
            </div>
            <div class="flex flex-wrap gap-2">
              <span class="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                通过 {{ resultQualityGates?.pass_count ?? 0 }}
              </span>
              <span class="px-2 py-0.5 rounded-full bg-rose-50 text-rose-700 border border-rose-200">
                失败 {{ resultQualityGates?.fail_count ?? 0 }}
              </span>
              <span class="px-2 py-0.5 rounded-full bg-slate-50 text-slate-700 border border-slate-200">
                跳过 {{ resultQualityGates?.skip_count ?? 0 }}
              </span>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
              <div
                v-for="check in resultQualityGates?.checks || []"
                :key="check.key"
                class="rounded-lg border px-2 py-1.5"
                :class="gateStatusClass(check.status)"
              >
                <div class="font-medium">{{ check.label }} · {{ gateStatusLabel(check.status) }}</div>
                <div class="text-[11px] mt-1">
                  当前 {{ formatNumber(check.value, 4) }} {{ check.operator }} 阈值 {{ formatNumber(check.threshold, 4) }}
                </div>
              </div>
            </div>
          </div>

          <div v-if="!resultIsClassifierOnly && resultReleaseDecision" class="rounded-xl border border-gray-100 bg-white p-3 space-y-3 text-xs">
            <div class="flex items-center justify-between">
              <div class="text-gray-700 font-semibold">发布决策</div>
              <span
                class="px-2 py-0.5 rounded-full border"
                :class="resultReleaseDecision?.blocked ? 'bg-rose-50 text-rose-700 border-rose-200' : 'bg-emerald-50 text-emerald-700 border-emerald-200'"
              >
                {{ releaseActionLabel(resultReleaseDecision?.action) }}
              </span>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-2 text-gray-600">
              <div>CI 模式：{{ resultReleaseDecision?.ci_mode ? '开启' : '关闭' }}</div>
              <div>阻断策略：{{ resultReleaseDecision?.effective_fail_on_gate ? '严格阻断' : '仅告警' }}</div>
            </div>
            <div class="text-gray-500">{{ resultReleaseDecision?.reason || '--' }}</div>
          </div>

          <div v-if="resultClassifierGate" class="rounded-xl border border-gray-100 bg-white p-3 space-y-3 text-xs">
            <div class="flex items-center justify-between">
              <div class="text-gray-700 font-semibold">分类器门禁</div>
              <span
                class="px-2 py-0.5 rounded-full border"
                :class="gateStatusClass(resultClassifierGate?.overall_status)"
              >
                {{ gateStatusLabel(resultClassifierGate?.overall_status) }}
              </span>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
              <div
                v-for="check in resultClassifierGate?.checks || []"
                :key="`classifier-gate-${check.key}`"
                class="rounded-lg border px-2 py-1.5"
                :class="gateStatusClass(check.status)"
              >
                <div class="font-medium">{{ check.label }} · {{ gateStatusLabel(check.status) }}</div>
                <div class="text-[11px] mt-1">
                  当前 {{ formatNumber(check.value, 4) }} {{ check.operator }} 阈值 {{ formatNumber(check.threshold, 4) }}
                </div>
              </div>
            </div>
          </div>

          <div v-if="!resultIsClassifierOnly && resultBaseline?.found" class="rounded-xl border border-gray-100 bg-white p-3 space-y-2 text-xs">
            <div class="text-gray-700 font-semibold">与历史基线对比</div>
            <div class="text-gray-500">
              基线 {{ formatHistoryTime(resultBaseline?.baseline_created_at) }} · {{ resultBaseline?.baseline_run_tag || '未命名' }}
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
              <div
                v-for="(deltaInfo, metricKey) in resultBaseline?.deltas || {}"
                :key="metricKey"
                class="rounded-lg border border-gray-200 bg-gray-50 px-2 py-1.5"
              >
                <div class="text-gray-600">{{ metricKey }}</div>
                <div class="text-gray-800">
                  Δ {{ formatDelta(deltaInfo?.delta) }}（{{ formatNumber(deltaInfo?.baseline, 4) }} -> {{ formatNumber(deltaInfo?.current, 4) }}）
                </div>
              </div>
            </div>
          </div>

          <div v-if="!resultIsClassifierOnly && resultSlices" class="rounded-xl border border-gray-100 bg-white p-3 space-y-3 text-xs">
            <div class="text-gray-700 font-semibold">分桶分析</div>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div v-for="(bucketMap, sliceKey) in resultSlices" :key="sliceKey" class="space-y-2">
                <div class="text-gray-500">{{ sliceKey }}</div>
                <div
                  v-for="(bucketStats, bucketName) in bucketMap || {}"
                  :key="`${sliceKey}-${bucketName}`"
                  class="rounded-lg border border-gray-200 bg-gray-50 px-2 py-1.5"
                >
                  <div class="text-gray-700 font-medium">{{ bucketName }} · {{ bucketStats?.count ?? 0 }}</div>
                  <div class="text-gray-600">
                    错误率 {{ formatPercent(bucketStats?.error_rate) }} · 转人工率 {{ formatPercent(bucketStats?.handoff_rate) }}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="space-y-3">
            <h3 class="text-sm font-semibold text-gray-900">详细样本</h3>
            <div v-if="!result" class="text-sm text-gray-500">
              运行评测后在此查看样本指标与检索上下文。
            </div>
            <div v-else class="space-y-4">
              <div
                v-if="!result.items || result.items.length === 0"
                class="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2"
              >
                本次结果未返回样本明细。评测记录已保存，可在下方“评测历史 -> 查看”中分页查看条目。
              </div>
              <div
                v-for="(item, index) in result.items"
                :key="index"
                class="border border-gray-200 rounded-xl p-4 bg-white space-y-3"
              >
                <div class="flex items-start justify-between gap-3">
                  <div>
                    <div class="text-xs text-gray-400">问题</div>
                    <div class="text-sm text-gray-900">{{ item.question }}</div>
                  </div>
                  <div class="text-xs text-gray-500 shrink-0">
                    上下文 {{ item.contexts_count || 0 }} · 延迟 {{ item.latency_ms ?? '--' }} ms
                  </div>
                </div>
                <div v-if="!resultIsClassifierOnly" class="grid gap-3">
                  <div>
                    <div class="text-xs text-gray-400">参考答案</div>
                    <div class="text-sm text-gray-700 whitespace-pre-wrap">{{ item.reference }}</div>
                  </div>
                  <div>
                    <div class="text-xs text-gray-400">模型回答</div>
                    <div class="text-sm text-gray-700 whitespace-pre-wrap">{{ item.answer }}</div>
                  </div>
                </div>
                <div v-if="resultIsClassifierOnly" class="rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-xs text-gray-700 space-y-2">
                  <div class="flex flex-wrap gap-2">
                    <span class="px-2 py-0.5 rounded-full bg-white border border-gray-200">
                      标注 {{ item.classifier_label_eval?.has_label ? formatBoolFlag(item.classifier_label_eval?.label) : '--' }}
                    </span>
                    <span class="px-2 py-0.5 rounded-full bg-white border border-gray-200">
                      预测 {{ formatBoolFlag(item.classifier_label_eval?.predicted) }}
                    </span>
                    <span
                      v-if="item.classifier_label_eval?.has_label"
                      class="px-2 py-0.5 rounded-full border"
                      :class="item.classifier_label_eval?.is_correct ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : 'bg-rose-50 text-rose-700 border-rose-200'"
                    >
                      {{ item.classifier_label_eval?.is_correct ? '判定正确' : '判定错误' }}
                    </span>
                    <span class="px-2 py-0.5 rounded-full bg-cyan-50 text-cyan-700 border border-cyan-200">
                      概率 {{ formatNumber(item.classifier_label_eval?.probability, 3) }}
                    </span>
                    <span
                      v-if="item.classifier_label_eval?.predicted_from_probability_fallback"
                      class="px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200"
                    >
                      概率回填决策
                    </span>
                  </div>
                  <div class="text-[11px] text-gray-500">
                    决策阶段 {{ formatRoutingStageLabel(item.routing?.stage) }} · 状态 {{ item.status === 'success' ? '成功' : '失败' }} ·
                    原因 {{ item.routing?.reason || item.error || '--' }}
                  </div>
                </div>
                <div v-if="!resultIsClassifierOnly" class="flex flex-wrap gap-2 text-xs">
                  <span
                    v-for="metric in metricDefinitions"
                    :key="metric.key"
                    class="px-2 py-0.5 rounded-full bg-gray-100 text-gray-600"
                  >
                    {{ metric.label }} {{ formatScore(getItemMetricValue(item, metric.key)) }}
                  </span>
                </div>
                <div v-if="!resultIsClassifierOnly" class="flex flex-wrap gap-2 text-xs">
                  <span
                    class="px-2 py-0.5 rounded-full border"
                    :class="routingStageClass(item.routing?.stage)"
                  >
                    路由 {{ formatRoutingStageLabel(item.routing?.stage) }}
                  </span>
                  <span class="px-2 py-0.5 rounded-full bg-gray-100 text-gray-600">
                    需检索 {{ formatBoolFlag(item.routing?.need_retrieval) }}
                  </span>
                  <span
                    v-if="item.routing?.statistical_classifier?.probability !== null && item.routing?.statistical_classifier?.probability !== undefined"
                    class="px-2 py-0.5 rounded-full bg-cyan-50 text-cyan-700 border border-cyan-200"
                  >
                    统计概率 {{ formatNumber(item.routing?.statistical_classifier?.probability, 3) }}
                  </span>
                  <span
                    v-if="item.classifier_label_eval?.has_label"
                    class="px-2 py-0.5 rounded-full border"
                    :class="item.classifier_label_eval?.is_correct ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : 'bg-rose-50 text-rose-700 border-rose-200'"
                  >
                    检索标签判定 {{ item.classifier_label_eval?.is_correct ? '正确' : '错误' }}
                  </span>
                  <span
                    v-for="tag in item.badcase?.tags || []"
                    :key="`curr-badcase-${index}-${tag}`"
                    class="px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200"
                  >
                    badcase {{ tag }}
                  </span>
                </div>
                <div v-if="!resultIsClassifierOnly" class="flex flex-wrap gap-2 text-xs">
                  <span class="px-2 py-0.5 rounded-full bg-sky-50 text-sky-700 border border-sky-200">
                    分诊 {{ formatTriageLabel(getItemSop(item).triage_level) }}
                  </span>
                  <span class="px-2 py-0.5 rounded-full bg-rose-50 text-rose-700 border border-rose-200">
                    转人工 {{ formatBoolFlag(getItemSop(item).handoff_required) }}
                  </span>
                  <span class="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                    结构化有效 {{ formatBoolFlag(getItemSop(item).structured_decision_valid) }}
                  </span>
                  <span
                    v-if="item.expected_handoff !== null && item.expected_handoff !== undefined"
                    class="px-2 py-0.5 rounded-full bg-violet-50 text-violet-700 border border-violet-200"
                  >
                    标注转人工 {{ formatBoolFlag(item.expected_handoff) }}
                  </span>
                </div>
                <details v-if="!resultIsClassifierOnly" class="bg-gray-50 rounded-lg p-3 text-xs text-gray-600">
                  <summary class="cursor-pointer text-gray-500">查看上下文预览</summary>
                  <div class="mt-2 space-y-2">
                    <div
                      v-for="(ctx, ctxIndex) in item.contexts_preview || []"
                      :key="ctxIndex"
                      class="p-2 rounded bg-white border border-gray-200"
                    >
                      {{ ctx }}
                    </div>
                    <div v-if="!item.contexts_preview || item.contexts_preview.length === 0">
                      未返回上下文
                    </div>
                  </div>
                </details>
                <details v-if="!resultIsClassifierOnly" class="bg-slate-50 rounded-lg p-3 text-xs text-gray-600">
                  <summary class="cursor-pointer text-gray-500">查看SOP执行链</summary>
                  <div class="mt-2 space-y-2">
                    <div>
                      <span class="text-gray-400">红线命中：</span>
                      <span class="text-gray-700">
                        {{ (getItemSop(item).red_flags || []).join('、') || '无' }}
                      </span>
                    </div>
                    <div>
                      <span class="text-gray-400">转人工原因：</span>
                      <span class="text-gray-700">{{ getItemSop(item).handoff_reason || '--' }}</span>
                    </div>
                    <div>
                      <span class="text-gray-400">症状抽取：</span>
                      <span class="text-gray-700">
                        {{ (getItemSop(item).symptoms || []).join('、') || '--' }}
                      </span>
                    </div>
                    <div>
                      <span class="text-gray-400">干预摘要：</span>
                      <span class="text-gray-700">{{ getItemSop(item).intervention_plan?.summary || '--' }}</span>
                    </div>
                    <div v-if="getItemSop(item).sop_steps">
                      <span class="text-gray-400">SOP步骤：</span>
                      <span class="text-gray-700">
                        <span
                          v-for="(stepValue, stepName) in getItemSop(item).sop_steps || {}"
                          :key="`sop-step-${stepName}`"
                          class="inline-block mr-2"
                        >
                          {{ stepName }}={{ stepValue?.passed ? 'pass' : 'fail' }}
                        </span>
                      </span>
                    </div>
                    <div
                      v-if="getItemSop(item).handoff_correct !== null && getItemSop(item).handoff_correct !== undefined"
                    >
                      <span class="text-gray-400">转人工判定正确：</span>
                      <span class="text-gray-700">{{ formatBoolFlag(getItemSop(item).handoff_correct) }}</span>
                    </div>
                  </div>
                </details>
                <details class="bg-slate-50 rounded-lg p-3 text-xs text-gray-600">
                  <summary class="cursor-pointer text-gray-500">查看路由决策链</summary>
                  <div class="mt-2 space-y-2">
                    <div>
                      <span class="text-gray-400">判定阶段：</span>
                      <span class="text-gray-700">{{ formatRoutingStageLabel(item.routing?.stage) }}</span>
                    </div>
                    <div>
                      <span class="text-gray-400">判定理由：</span>
                      <span class="text-gray-700">{{ item.routing?.reason || '--' }}</span>
                    </div>
                    <div>
                      <span class="text-gray-400">统计分类器：</span>
                      <span class="text-gray-700">
                        p={{ formatNumber(item.routing?.statistical_classifier?.probability, 3) }}，决策={{ formatBoolFlag(item.routing?.statistical_classifier?.decision) }}
                      </span>
                    </div>
                    <div>
                      <span class="text-gray-400">决策路径尾部：</span>
                      <span class="text-gray-700">{{ (item.routing?.decision_path_tail || []).join(' -> ') || '--' }}</span>
                    </div>
                  </div>
                </details>
                <details v-if="!resultIsClassifierOnly" class="bg-slate-50 rounded-lg p-3 text-xs text-gray-600">
                  <summary class="cursor-pointer text-gray-500">查看生成-证据对齐</summary>
                  <div class="mt-2 space-y-2">
                    <div class="text-gray-500">
                      完整性 {{ formatPercent(item.generation_alignment?.completeness) }} · 引用覆盖 {{ formatPercent(item.generation_alignment?.citation_coverage) }}
                    </div>
                    <div
                      v-for="(align, alignIndex) in item.generation_alignment?.alignments || []"
                      :key="`align-${alignIndex}`"
                      class="rounded bg-white border border-gray-200 px-2 py-1.5"
                    >
                      <div class="text-gray-700">{{ align.sentence }}</div>
                      <div class="text-[11px] text-gray-500">
                        support={{ formatNumber(align.support_score, 3) }} · ctx={{ align.best_context_index ?? '--' }} · {{ align.supported ? 'supported' : 'unsupported' }}
                      </div>
                    </div>
                    <div v-if="!(item.generation_alignment?.alignments || []).length" class="text-gray-500">
                      暂无可对齐句子
                    </div>
                  </div>
                </details>
                <details v-if="!resultIsClassifierOnly" class="bg-amber-50 rounded-lg p-3 text-xs text-amber-700 border border-amber-200">
                  <summary class="cursor-pointer">查看Badcase诊断</summary>
                  <div class="mt-2 space-y-2">
                    <div>
                      <span class="text-amber-500">严重级别：</span>
                      <span>{{ item.badcase?.severity || 'none' }}</span>
                    </div>
                    <div>
                      <span class="text-amber-500">主标签：</span>
                      <span>{{ item.badcase?.primary_tag || '--' }}</span>
                    </div>
                    <div>
                      <span class="text-amber-500">诊断说明：</span>
                      <span>{{ item.badcase?.reason || '--' }}</span>
                    </div>
                  </div>
                </details>
              </div>
            </div>
          </div>
        </BaseCard>
      </div>

      <BaseCard class="space-y-4">
        <div class="flex items-center justify-between">
          <h2 class="text-lg font-semibold text-gray-900">评测历史</h2>
          <div class="flex items-center gap-3 text-xs text-gray-500">
            <span>最近 50 条</span>
            <button class="text-gray-500 hover:text-gray-700" @click="loadHistories">刷新</button>
          </div>
        </div>
        <div v-if="historyLoading" class="text-sm text-gray-500">加载中...</div>
        <div v-else-if="histories.length === 0" class="text-sm text-gray-500">暂无历史记录</div>
        <div v-else class="space-y-3">
          <div
            v-for="item in histories"
            :key="item.id"
            class="border border-gray-200 rounded-xl p-4 bg-white space-y-3"
          >
            <div class="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div class="text-sm font-semibold text-gray-900">
                  {{ item.dataset_name || '未命名数据集' }}
                </div>
                <div class="text-xs text-gray-500 mt-1">
                  {{ formatHistoryTime(item.created_at) }}
                  · {{ getHistoryTotal(item) }} 条
                  · {{ item.retrieval_mode || '--' }}
                </div>
                <div class="mt-2 flex flex-wrap items-center gap-2 text-[11px]">
                  <span class="px-2 py-0.5 rounded-full border bg-slate-50 text-slate-700 border-slate-200">
                    模块 {{ formatEvalScopeLabel(getHistoryEvalScope(item)) }}
                  </span>
                  <span
                    v-if="getHistoryReleaseAction(item)"
                    class="px-2 py-0.5 rounded-full border"
                    :class="releaseActionClass(getHistoryReleaseAction(item))"
                  >
                    {{ releaseActionLabel(getHistoryReleaseAction(item)) }}
                  </span>
                </div>
              </div>
              <div class="flex items-center gap-2">
                <BaseButton
                  size="sm"
                  variant="secondary"
                  :loading="historyDetailLoading && selectedHistoryId === item.id"
                  @click="viewHistory(item.id)"
                >
                  查看
                </BaseButton>
                <BaseButton size="sm" variant="ghost" @click="applyHistoryResult(item)">
                  加载到结果
                </BaseButton>
              </div>
            </div>
            <div class="flex flex-wrap gap-2 text-xs text-gray-600">
              <span
                v-for="metric in metricDefinitions"
                :key="metric.key"
                class="px-2 py-0.5 rounded-full bg-gray-100 text-gray-600"
              >
                {{ metric.label }} {{ formatScore(getHistorySummaryValue(item, metric.key)) }}
              </span>
            </div>
          </div>
        </div>
        <div v-if="historyDetailLoading" class="text-sm text-gray-500">历史详情加载中...</div>
        <div v-else-if="historyDetail" class="border-t border-gray-100 pt-4 space-y-3">
          <div class="flex items-center justify-between">
            <div class="text-sm font-semibold text-gray-900">历史详情</div>
            <div class="flex items-center gap-2">
              <BaseButton size="sm" variant="secondary" @click="exportHistoryReport">
                导出历史实验室
              </BaseButton>
              <BaseButton size="sm" variant="secondary" @click="applyHistoryResult(historyDetail)">
                加载到结果
              </BaseButton>
            </div>
          </div>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs text-gray-500">
            <div>数据集：{{ historyDetail.dataset_name || '未命名数据集' }}</div>
            <div>条数：{{ getHistoryTotal(historyDetail) }}</div>
            <div>模块：{{ formatEvalScopeLabel(getHistoryEvalScope(historyDetail)) }}</div>
            <div>检索模式：{{ historyDetail.retrieval_mode || '--' }}</div>
            <div>工作区：{{ historyDetail.workspace || '--' }}</div>
            <div v-if="historyDetail.collection_id">知识库：{{ historyDetail.collection_id }}</div>
            <div>用时：{{ getHistoryElapsed(historyDetail) }} ms</div>
          </div>
          <div v-if="getHistoryReleaseAction(historyDetail)" class="text-xs">
            <span class="px-2 py-0.5 rounded-full border" :class="releaseActionClass(getHistoryReleaseAction(historyDetail))">
              {{ releaseActionLabel(getHistoryReleaseAction(historyDetail)) }}
            </span>
          </div>

          <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <div
              v-for="metric in metricDefinitions"
              :key="`history-${metric.key}`"
              class="rounded-xl border border-gray-100 bg-slate-50 p-3"
            >
              <div class="text-gray-500">{{ metric.label }}</div>
              <div class="mt-2 text-base font-semibold text-gray-900">
                {{ formatScore(historyDetailSummary?.[metric.key]) }}
              </div>
            </div>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-6 gap-3 text-xs">
            <div class="rounded-xl border border-gray-100 bg-slate-50 p-3">
              <div class="text-gray-500">拒答率</div>
              <div class="mt-2 text-base font-semibold text-gray-900">
                {{ formatPercent(historyDetailStability?.abstained_answer_rate) }}
              </div>
            </div>
            <div class="rounded-xl border border-gray-100 bg-slate-50 p-3">
              <div class="text-gray-500">错误数</div>
              <div class="mt-2 text-base font-semibold text-gray-900">
                {{ historyItemsStats?.error_count ?? '--' }}
              </div>
            </div>
            <div class="rounded-xl border border-gray-100 bg-slate-50 p-3">
              <div class="text-gray-500">平均延迟</div>
              <div class="mt-2 text-base font-semibold text-gray-900">
                {{ formatNumber(historyItemsStats?.avg_latency_ms) }} ms
              </div>
            </div>
            <div class="rounded-xl border border-gray-100 bg-slate-50 p-3">
              <div class="text-gray-500">平均上下文条数</div>
              <div class="mt-2 text-base font-semibold text-gray-900">
                {{ formatNumber(historyItemsStats?.avg_contexts_count) }}
              </div>
            </div>
            <div class="rounded-xl border border-gray-100 bg-slate-50 p-3">
              <div class="text-gray-500">历史转人工率</div>
              <div class="mt-2 text-base font-semibold text-gray-900">
                {{ formatPercent(historyDetailSop?.handoff_rate) }}
              </div>
            </div>
            <div class="rounded-xl border border-gray-100 bg-slate-50 p-3">
              <div class="text-gray-500">历史LLM兜底率</div>
              <div class="mt-2 text-base font-semibold text-gray-900">
                {{ formatPercent(historyDetailRouting?.llm_fallback_rate) }}
              </div>
            </div>
            <div class="rounded-xl border border-gray-100 bg-slate-50 p-3">
              <div class="text-gray-500">分类器标注样本</div>
              <div class="mt-2 text-base font-semibold text-gray-900">
                {{ historyDetailClassifier?.labeled_items ?? historyItemsStats?.classifier_labeled_count ?? '--' }}
              </div>
            </div>
            <div class="rounded-xl border border-gray-100 bg-slate-50 p-3">
              <div class="text-gray-500">分类器准确率</div>
              <div class="mt-2 text-base font-semibold text-gray-900">
                {{ formatPercent(historyDetailClassifier?.accuracy ?? historyItemsStats?.classifier_accuracy) }}
              </div>
            </div>
            <div class="rounded-xl border border-gray-100 bg-slate-50 p-3">
              <div class="text-gray-500">Badcase 数量</div>
              <div class="mt-2 text-base font-semibold text-gray-900">
                {{ historyDetailBadcase?.badcase_count ?? historyItemsStats?.badcase_count ?? '--' }}
              </div>
            </div>
            <div class="rounded-xl border border-gray-100 bg-slate-50 p-3">
              <div class="text-gray-500">Badcase 占比</div>
              <div class="mt-2 text-base font-semibold text-gray-900">
                {{ formatPercent(historyDetailBadcase?.badcase_rate ?? historyItemsStats?.badcase_rate) }}
              </div>
            </div>
          </div>

          <div v-if="historyDetailGates" class="rounded-xl border border-gray-100 bg-white p-3 space-y-2 text-xs">
            <div class="flex items-center justify-between">
              <div class="text-gray-700 font-semibold">历史质量门禁</div>
              <span
                class="px-2 py-0.5 rounded-full border"
                :class="gateStatusClass(historyDetailGates?.overall_status)"
              >
                {{ gateStatusLabel(historyDetailGates?.overall_status) }}
              </span>
            </div>
            <div class="text-gray-500">
              通过 {{ historyDetailGates?.pass_count ?? 0 }} · 失败 {{ historyDetailGates?.fail_count ?? 0 }} · 跳过 {{ historyDetailGates?.skip_count ?? 0 }}
            </div>
          </div>

          <div v-if="historyDetailBaseline?.found" class="rounded-xl border border-gray-100 bg-white p-3 text-xs text-gray-600">
            历史记录已关联基线：{{ historyDetailBaseline?.baseline_id || '--' }}
          </div>

          <div v-if="historyDetailReproducibility" class="rounded-xl border border-gray-100 bg-white p-3 text-xs text-gray-600">
            <div>Replay Key：{{ historyDetailReproducibility?.replay_key || '--' }}</div>
            <div>Dataset Hash：{{ formatHashShort(historyDetailReproducibility?.dataset_sha256) }}</div>
            <div>Config Hash：{{ formatHashShort(historyDetailReproducibility?.config_sha256) }}</div>
          </div>

          <div class="space-y-3">
            <div class="flex items-center justify-between">
              <div class="text-sm font-semibold text-gray-900">历史样本明细</div>
              <div class="text-xs text-gray-500">
                共 {{ historyItemsTotal }} 条 · 第 {{ historyItemsPage }} / {{ historyItemsTotalPages }} 页
              </div>
            </div>
            <div v-if="historyItemsStats" class="grid grid-cols-2 md:grid-cols-7 gap-2 text-xs">
              <div class="rounded-lg border border-gray-200 bg-gray-50 px-2 py-1.5 text-gray-700">
                转人工 {{ historyItemsStats.handoff_count ?? '--' }}
              </div>
              <div class="rounded-lg border border-gray-200 bg-gray-50 px-2 py-1.5 text-gray-700">
                急诊 {{ historyItemsStats.emergency_count ?? '--' }}
              </div>
              <div class="rounded-lg border border-gray-200 bg-gray-50 px-2 py-1.5 text-gray-700">
                转人工率 {{ formatPercent(historyItemsStats.handoff_rate) }}
              </div>
              <div class="rounded-lg border border-gray-200 bg-gray-50 px-2 py-1.5 text-gray-700">
                标注准确率 {{ formatPercent(historyItemsStats.handoff_accuracy) }}
              </div>
              <div class="rounded-lg border border-gray-200 bg-gray-50 px-2 py-1.5 text-gray-700">
                LLM兜底率 {{ formatPercent(historyItemsStats.routing_llm_fallback_rate) }}
              </div>
              <div class="rounded-lg border border-gray-200 bg-gray-50 px-2 py-1.5 text-gray-700">
                分类器准确率 {{ formatPercent(historyItemsStats.classifier_accuracy) }}
              </div>
              <div class="rounded-lg border border-gray-200 bg-gray-50 px-2 py-1.5 text-gray-700">
                分类器不确定率 {{ formatPercent(historyItemsStats.classifier_uncertain_rate) }}
              </div>
            </div>
            <div v-if="historyItemsLoading" class="text-sm text-gray-500">样本加载中...</div>
            <div v-else-if="!historyItemsAvailable" class="text-sm text-gray-500">该历史记录没有样本详情</div>
            <div v-else class="space-y-3">
              <div
                v-for="(item, index) in historyItems"
                :key="`${historyItemsPage}-${index}`"
                class="border border-gray-200 rounded-xl p-4 bg-white space-y-3"
              >
                <div class="flex items-start justify-between gap-3">
                  <div>
                    <div class="text-xs text-gray-400">问题</div>
                    <div class="text-sm text-gray-900">{{ item.question }}</div>
                  </div>
                  <div class="text-xs text-gray-500 shrink-0">
                    上下文 {{ item.contexts_count || 0 }} · 延迟 {{ item.latency_ms ?? '--' }} ms
                  </div>
                </div>
                <div class="grid gap-3">
                  <div>
                    <div class="text-xs text-gray-400">参考答案</div>
                    <div class="text-sm text-gray-700 whitespace-pre-wrap">{{ item.reference }}</div>
                  </div>
                  <div>
                    <div class="text-xs text-gray-400">模型回答</div>
                    <div class="text-sm text-gray-700 whitespace-pre-wrap">{{ item.answer }}</div>
                  </div>
                </div>
                <div class="flex flex-wrap gap-2 text-xs">
                  <span
                    v-for="metric in metricDefinitions"
                    :key="metric.key"
                    class="px-2 py-0.5 rounded-full bg-gray-100 text-gray-600"
                  >
                    {{ metric.label }} {{ formatScore(getItemMetricValue(item, metric.key)) }}
                  </span>
                </div>
                <div class="flex flex-wrap gap-2 text-xs">
                  <span
                    class="px-2 py-0.5 rounded-full border"
                    :class="routingStageClass(item.routing?.stage)"
                  >
                    路由 {{ formatRoutingStageLabel(item.routing?.stage) }}
                  </span>
                  <span class="px-2 py-0.5 rounded-full bg-gray-100 text-gray-600">
                    需检索 {{ formatBoolFlag(item.routing?.need_retrieval) }}
                  </span>
                  <span
                    v-if="item.routing?.statistical_classifier?.probability !== null && item.routing?.statistical_classifier?.probability !== undefined"
                    class="px-2 py-0.5 rounded-full bg-cyan-50 text-cyan-700 border border-cyan-200"
                  >
                    统计概率 {{ formatNumber(item.routing?.statistical_classifier?.probability, 3) }}
                  </span>
                  <span
                    v-if="item.classifier_label_eval?.has_label"
                    class="px-2 py-0.5 rounded-full border"
                    :class="item.classifier_label_eval?.is_correct ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : 'bg-rose-50 text-rose-700 border-rose-200'"
                  >
                    检索标签判定 {{ item.classifier_label_eval?.is_correct ? '正确' : '错误' }}
                  </span>
                  <span
                    v-for="tag in item.badcase?.tags || []"
                    :key="`hist-badcase-${historyItemsPage}-${index}-${tag}`"
                    class="px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200"
                  >
                    badcase {{ tag }}
                  </span>
                </div>
                <div class="flex flex-wrap gap-2 text-xs">
                  <span class="px-2 py-0.5 rounded-full bg-sky-50 text-sky-700 border border-sky-200">
                    分诊 {{ formatTriageLabel(getItemSop(item).triage_level) }}
                  </span>
                  <span class="px-2 py-0.5 rounded-full bg-rose-50 text-rose-700 border border-rose-200">
                    转人工 {{ formatBoolFlag(getItemSop(item).handoff_required) }}
                  </span>
                  <span
                    v-if="item.expected_handoff !== null && item.expected_handoff !== undefined"
                    class="px-2 py-0.5 rounded-full bg-violet-50 text-violet-700 border border-violet-200"
                  >
                    标注转人工 {{ formatBoolFlag(item.expected_handoff) }}
                  </span>
                </div>
                <details class="bg-gray-50 rounded-lg p-3 text-xs text-gray-600">
                  <summary class="cursor-pointer text-gray-500">查看上下文预览</summary>
                  <div class="mt-2 space-y-2">
                    <div
                      v-for="(ctx, ctxIndex) in item.contexts_preview || []"
                      :key="ctxIndex"
                      class="p-2 rounded bg-white border border-gray-200"
                    >
                      {{ ctx }}
                    </div>
                    <div v-if="!item.contexts_preview || item.contexts_preview.length === 0">
                      未返回上下文
                    </div>
                  </div>
                </details>
                <details class="bg-slate-50 rounded-lg p-3 text-xs text-gray-600">
                  <summary class="cursor-pointer text-gray-500">查看SOP执行链</summary>
                  <div class="mt-2 space-y-2">
                    <div>
                      <span class="text-gray-400">红线命中：</span>
                      <span class="text-gray-700">
                        {{ (getItemSop(item).red_flags || []).join('、') || '无' }}
                      </span>
                    </div>
                    <div>
                      <span class="text-gray-400">转人工原因：</span>
                      <span class="text-gray-700">{{ getItemSop(item).handoff_reason || '--' }}</span>
                    </div>
                    <div>
                      <span class="text-gray-400">干预摘要：</span>
                      <span class="text-gray-700">{{ getItemSop(item).intervention_plan?.summary || '--' }}</span>
                    </div>
                    <div v-if="getItemSop(item).sop_steps">
                      <span class="text-gray-400">SOP步骤：</span>
                      <span class="text-gray-700">
                        <span
                          v-for="(stepValue, stepName) in getItemSop(item).sop_steps || {}"
                          :key="`hist-sop-step-${stepName}`"
                          class="inline-block mr-2"
                        >
                          {{ stepName }}={{ stepValue?.passed ? 'pass' : 'fail' }}
                        </span>
                      </span>
                    </div>
                  </div>
                </details>
                <details class="bg-slate-50 rounded-lg p-3 text-xs text-gray-600">
                  <summary class="cursor-pointer text-gray-500">查看路由决策链</summary>
                  <div class="mt-2 space-y-2">
                    <div>
                      <span class="text-gray-400">判定阶段：</span>
                      <span class="text-gray-700">{{ formatRoutingStageLabel(item.routing?.stage) }}</span>
                    </div>
                    <div>
                      <span class="text-gray-400">判定理由：</span>
                      <span class="text-gray-700">{{ item.routing?.reason || '--' }}</span>
                    </div>
                    <div>
                      <span class="text-gray-400">统计分类器：</span>
                      <span class="text-gray-700">
                        p={{ formatNumber(item.routing?.statistical_classifier?.probability, 3) }}，决策={{ formatBoolFlag(item.routing?.statistical_classifier?.decision) }}
                      </span>
                    </div>
                  </div>
                </details>
                <details class="bg-slate-50 rounded-lg p-3 text-xs text-gray-600">
                  <summary class="cursor-pointer text-gray-500">查看生成-证据对齐</summary>
                  <div class="mt-2 space-y-2">
                    <div class="text-gray-500">
                      完整性 {{ formatPercent(item.generation_alignment?.completeness) }} · 引用覆盖 {{ formatPercent(item.generation_alignment?.citation_coverage) }}
                    </div>
                    <div
                      v-for="(align, alignIndex) in item.generation_alignment?.alignments || []"
                      :key="`hist-align-${alignIndex}`"
                      class="rounded bg-white border border-gray-200 px-2 py-1.5"
                    >
                      <div class="text-gray-700">{{ align.sentence }}</div>
                      <div class="text-[11px] text-gray-500">
                        support={{ formatNumber(align.support_score, 3) }} · ctx={{ align.best_context_index ?? '--' }} · {{ align.supported ? 'supported' : 'unsupported' }}
                      </div>
                    </div>
                    <div v-if="!(item.generation_alignment?.alignments || []).length" class="text-gray-500">
                      暂无可对齐句子
                    </div>
                  </div>
                </details>
                <details class="bg-amber-50 rounded-lg p-3 text-xs text-amber-700 border border-amber-200">
                  <summary class="cursor-pointer">查看Badcase诊断</summary>
                  <div class="mt-2 space-y-2">
                    <div>
                      <span class="text-amber-500">严重级别：</span>
                      <span>{{ item.badcase?.severity || 'none' }}</span>
                    </div>
                    <div>
                      <span class="text-amber-500">主标签：</span>
                      <span>{{ item.badcase?.primary_tag || '--' }}</span>
                    </div>
                    <div>
                      <span class="text-amber-500">诊断说明：</span>
                      <span>{{ item.badcase?.reason || '--' }}</span>
                    </div>
                  </div>
                </details>
              </div>
            </div>
            <div v-if="historyItemsAvailable" class="flex items-center justify-end gap-2">
              <BaseButton
                size="sm"
                variant="secondary"
                :disabled="historyItemsPage <= 1 || historyItemsLoading"
                @click="changeHistoryItemsPage(historyItemsPage - 1)"
              >
                上一页
              </BaseButton>
              <BaseButton
                size="sm"
                variant="secondary"
                :disabled="historyItemsPage >= historyItemsTotalPages || historyItemsLoading"
                @click="changeHistoryItemsPage(historyItemsPage + 1)"
              >
                下一页
              </BaseButton>
            </div>
          </div>
        </div>
      </BaseCard>
    </div>
  </div>
</template>
