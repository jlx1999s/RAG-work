import { httpClient } from './config.js'

export async function runRagEvaluation(payload) {
  return httpClient.postWithOptions('/rag/evaluate', payload, { timeout: 180000 })
}

export async function startRagEvaluationAsync(payload) {
  return httpClient.post('/rag/evaluate-async', payload)
}

export async function getRagEvaluationStatus(taskId) {
  return httpClient.get(`/rag/evaluate-status/${encodeURIComponent(taskId)}`)
}

export async function cancelRagEvaluation(taskId) {
  return httpClient.post(`/rag/evaluate-cancel/${encodeURIComponent(taskId)}`, {})
}

export async function trainIntentClassifier(payload) {
  return httpClient.post('/rag/classifier-train', payload)
}

export async function listEvalDatasets() {
  return httpClient.get('/rag/eval-datasets')
}

export async function getEvalDatasetContent(datasetName) {
  return httpClient.get(`/rag/eval-datasets/${encodeURIComponent(datasetName)}`)
}

export async function saveEvalDataset(payload) {
  return httpClient.post('/rag/eval-datasets', payload)
}

export async function listEvalHistory(params = {}) {
  const query = new URLSearchParams(params).toString()
  const suffix = query ? `?${query}` : ''
  return httpClient.get(`/rag/eval-history${suffix}`)
}

export async function getEvalHistory(historyId, params = {}) {
  const query = new URLSearchParams(params).toString()
  const suffix = query ? `?${query}` : ''
  return httpClient.get(`/rag/eval-history/${encodeURIComponent(historyId)}${suffix}`)
}

export async function getEvalHistoryItems(historyId, params = {}) {
  const query = new URLSearchParams(params).toString()
  const suffix = query ? `?${query}` : ''
  return httpClient.get(`/rag/eval-history/${encodeURIComponent(historyId)}/items${suffix}`)
}

export async function getEvalHistoryBadcases(historyId, params = {}) {
  const query = new URLSearchParams(params).toString()
  const suffix = query ? `?${query}` : ''
  return httpClient.get(`/rag/eval-history/${encodeURIComponent(historyId)}/badcases${suffix}`)
}
