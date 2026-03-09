import { httpClient } from './config.js'

export async function runRagEvaluation(payload) {
  return httpClient.postWithOptions('/rag/evaluate', payload, { timeout: 180000 })
}

export async function startRagEvaluationAsync(payload) {
  return httpClient.post('/rag/evaluate-async', payload)
}

export async function getRagEvaluationStatus(taskId) {
  return httpClient.get(`/rag/evaluate-status/${taskId}`)
}

export async function listEvalDatasets() {
  return httpClient.get('/rag/eval-datasets')
}

export async function getEvalDatasetContent(datasetName) {
  return httpClient.get(`/rag/eval-datasets/${datasetName}`)
}

export async function saveEvalDataset(payload) {
  return httpClient.post('/rag/eval-datasets', payload)
}

export async function listEvalHistory(params = {}) {
  const query = new URLSearchParams(params).toString()
  const suffix = query ? `?${query}` : ''
  return httpClient.get(`/rag/eval-history${suffix}`)
}

export async function getEvalHistory(historyId) {
  return httpClient.get(`/rag/eval-history/${historyId}`)
}
