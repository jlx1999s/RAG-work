<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import BaseButton from '@/components/BaseButton.vue'
import BaseCard from '@/components/BaseCard.vue'
import { knowledgeAPI } from '@/api/knowledge.js'
import {
  getEvalDatasetContent,
  getRagEvaluationStatus,
  getEvalHistory,
  getEvalHistoryItems,
  listEvalHistory,
  listEvalDatasets,
  saveEvalDataset,
  startRagEvaluationAsync
} from '@/api/evaluation.js'

const router = useRouter()

const sampleDataset = `{"question":"RAG系统的主要目标是什么？","reference":"RAG系统的主要目标是结合检索与生成，在回答中利用外部知识以提高准确性与可解释性。"}
{"question":"向量检索通常使用什么表示文本？","reference":"向量检索通常使用文本的向量表示，例如由嵌入模型生成的稠密向量。"}
{"question":"图检索更擅长处理哪类信息？","reference":"图检索更擅长处理实体关系与结构化关联信息。"}`

const datasetText = ref(sampleDataset)
const limit = ref(10)
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
const libraries = ref([])
const datasets = ref([])
const selectedDataset = ref('sample')
const datasetLoading = ref(false)
const loading = ref(false)
const saving = ref(false)
const result = ref(null)
const taskId = ref('')
const polling = ref(false)
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

const resultStability = computed(() => {
  return result.value?.run?.stability_summary ?? result.value?.stability_summary ?? null
})

const resultCost = computed(() => {
  return result.value?.run?.cost_summary ?? result.value?.cost_summary ?? null
})

const resultCache = computed(() => {
  return result.value?.run?.cache_summary ?? result.value?.cache_summary ?? null
})

const historyDetailSummary = computed(() => {
  return historyDetail.value?.result?.run?.quality_summary ?? historyDetail.value?.result?.summary ?? null
})

const historyDetailStability = computed(() => {
  return historyDetail.value?.result?.run?.stability_summary ?? historyDetail.value?.result?.stability_summary ?? null
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
      size: null
    }
  }
  return datasets.value.find((item) => item.name === selectedDataset.value) || null
})

const datasetExists = computed(() => {
  const name = datasetName.value.trim()
  if (!name) return false
  const fileName = name.toLowerCase().endsWith('.jsonl') ? name : `${name}.jsonl`
  return datasets.value.some((item) => item.name === fileName)
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
  polling.value = false
  result.value = null
  try {
    const response = await startRagEvaluationAsync({
      dataset_jsonl: datasetText.value,
      limit: Number(limit.value) || 0,
      workspace: workspace.value || 'eval_ws',
      retrieval_mode: retrievalMode.value,
      max_retrieval_docs: Number(maxDocs.value) || 3,
      collection_id: collectionId.value || null,
      run_tag: runTag.value.trim() || null,
      enable_ragas: Boolean(enableRagas.value),
      ragas_limit: Number(ragasLimit.value) || 0,
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
          if (statusPayload.status === 'completed') {
            result.value = statusPayload.result
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
            polling.value = false
            loading.value = false
            if (pollTimer) {
              clearInterval(pollTimer)
              pollTimer = null
            }
            ElMessage.error(statusPayload.error || '评测失败')
          }
        } catch (error) {
          polling.value = false
          loading.value = false
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

onMounted(() => {
  loadLibraries()
  loadDatasets()
  loadHistories()
})

watch(
  () => selectedDataset.value,
  (value) => {
    if (value && value !== 'sample' && !datasetName.value.trim()) {
      datasetName.value = value.replace(/\.jsonl$/i, '')
    }
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
            一键运行RAG评测，查看全局指标与单条样本结果
          </p>
        </div>
        <div class="flex items-center gap-3">
          <BaseButton variant="secondary" size="sm" @click="goBack">
            返回
          </BaseButton>
          <div class="px-3 py-1.5 rounded-full bg-white border border-gray-200 text-xs text-gray-600">
            数据条数 {{ datasetCount }}
          </div>
          <BaseButton :loading="loading" variant="primary" @click="runEvaluation">
            {{ loading ? '评测中...' : '运行评测' }}
          </BaseButton>
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
              <label class="text-xs font-medium text-gray-500">知识库</label>
              <select v-model="collectionId" class="input mt-2">
                <option value="">默认知识库</option>
                <option v-for="library in libraries" :key="library.id" :value="library.collection_id">
                  {{ library.title }}
                </option>
              </select>
            </div>
            <div>
              <label class="text-xs font-medium text-gray-500">检索模式</label>
              <select v-model="retrievalMode" class="input mt-2">
                <option value="vector_only">向量检索</option>
                <option value="hybrid">融合检索</option>
                <option value="graph_only">图检索</option>
                <option value="auto">自动选择</option>
                <option value="no_retrieval">不检索</option>
              </select>
            </div>
            <div>
              <label class="text-xs font-medium text-gray-500">最大检索文档</label>
              <input v-model="maxDocs" type="number" min="1" class="input mt-2" />
            </div>
            <div>
              <label class="text-xs font-medium text-gray-500">评测条数上限</label>
              <input v-model="limit" type="number" min="0" class="input mt-2" />
            </div>
            <div class="md:col-span-2">
              <label class="text-xs font-medium text-gray-500">评测工作区</label>
              <select v-model="workspace" class="input mt-2">
                <option v-for="option in workspaceOptions" :key="option.value" :value="option.value">
                  {{ option.label }}
                </option>
              </select>
              <div class="text-[11px] text-gray-400 mt-2">
                用于隔离评测运行的存储空间，与评测数据集无关
              </div>
            </div>
            <div>
              <label class="text-xs font-medium text-gray-500">运行标签</label>
              <input v-model="runTag" type="text" class="input mt-2" placeholder="例如：med-v2-baseline" />
            </div>
            <div>
              <label class="text-xs font-medium text-gray-500">RAGAS采样条数</label>
              <input
                v-model="ragasLimit"
                type="number"
                min="0"
                class="input mt-2"
                placeholder="0表示全量评测"
              />
              <div class="text-[11px] text-gray-400 mt-2">
                0 为全量；默认 10 条可显著缩短评测时长
              </div>
            </div>
            <div>
              <label class="text-xs font-medium text-gray-500">缓存命名空间</label>
              <input v-model="cacheNamespace" type="text" class="input mt-2" placeholder="例如：eval-default" />
            </div>
            <div class="md:col-span-2 grid grid-cols-1 md:grid-cols-3 gap-3">
              <label class="flex items-center gap-2 text-xs text-gray-600">
                <input v-model="enableRagas" type="checkbox" class="h-4 w-4" />
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
                    {{ dataset.name }} · {{ dataset.lines || 0 }}条
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
            <div class="text-xs text-gray-500" v-if="result">
              共 {{ resultTotal }} 条，用时 {{ resultElapsedMs }} ms
            </div>
          </div>

          <div class="grid grid-cols-2 gap-4">
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

          <div v-if="resultPerformance || resultRetrieval || resultStability || resultCost || resultCache" class="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
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
              </div>
              <div class="flex flex-wrap gap-2 text-[11px] text-gray-600">
                <span v-for="[sourceName, count] in sourceEntries" :key="sourceName" class="px-2 py-0.5 rounded-full bg-white border border-gray-200">
                  {{ sourceName }} {{ count }}
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
          </div>

          <div class="space-y-3">
            <h3 class="text-sm font-semibold text-gray-900">详细样本</h3>
            <div v-if="!result" class="text-sm text-gray-500">
              运行评测后在此查看样本指标与检索上下文。
            </div>
            <div v-else class="space-y-4">
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
            <BaseButton size="sm" variant="secondary" @click="applyHistoryResult(historyDetail)">
              加载到结果
            </BaseButton>
          </div>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs text-gray-500">
            <div>数据集：{{ historyDetail.dataset_name || '未命名数据集' }}</div>
            <div>条数：{{ getHistoryTotal(historyDetail) }}</div>
            <div>检索模式：{{ historyDetail.retrieval_mode || '--' }}</div>
            <div>工作区：{{ historyDetail.workspace || '--' }}</div>
            <div v-if="historyDetail.collection_id">知识库：{{ historyDetail.collection_id }}</div>
            <div>用时：{{ getHistoryElapsed(historyDetail) }} ms</div>
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

          <div class="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
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
          </div>

          <div class="space-y-3">
            <div class="flex items-center justify-between">
              <div class="text-sm font-semibold text-gray-900">历史样本明细</div>
              <div class="text-xs text-gray-500">
                共 {{ historyItemsTotal }} 条 · 第 {{ historyItemsPage }} / {{ historyItemsTotalPages }} 页
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
