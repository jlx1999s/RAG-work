/**
 * 知识库API服务模块
 * 提供知识库和文档的增删改查功能
 */

import { httpClient } from './config.js'

/**
 * 知识库API服务类
 */
class KnowledgeLibraryAPI {
  
  /**
   * 获取用户的知识库列表
   * @returns {Promise} API响应
   */
  async getLibraries() {
    try {
      const response = await httpClient.get('/api/knowledge/libraries')
      return response
    } catch (error) {
      console.error('获取知识库列表失败:', error)
      throw error
    }
  }

  /**
   * 获取知识库详情
   * @param {number} libraryId - 知识库ID
   * @returns {Promise} API响应
   */
  async getLibraryDetail(libraryId) {
    try {
      const response = await httpClient.get(`/api/knowledge/libraries/${libraryId}`)
      return response
    } catch (error) {
      console.error('获取知识库详情失败:', error)
      throw error
    }
  }

  /**
   * 创建知识库
   * @param {Object} libraryData - 知识库数据
   * @param {string} libraryData.title - 知识库标题
   * @param {string} libraryData.description - 知识库描述
   * @returns {Promise} API响应
   */
  async createLibrary(libraryData) {
    try {
      const response = await httpClient.post('/api/knowledge/libraries', libraryData)
      return response
    } catch (error) {
      console.error('创建知识库失败:', error)
      throw error
    }
  }

  /**
   * 更新知识库
   * @param {number} libraryId - 知识库ID
   * @param {Object} libraryData - 知识库数据
   * @returns {Promise} API响应
   */
  async updateLibrary(libraryId, libraryData) {
    try {
      const response = await httpClient.put(`/api/knowledge/libraries/${libraryId}`, libraryData)
      return response
    } catch (error) {
      console.error('更新知识库失败:', error)
      throw error
    }
  }

  /**
   * 删除知识库
   * @param {number} libraryId - 知识库ID
   * @returns {Promise} API响应
   */
  async deleteLibrary(libraryId) {
    try {
      const response = await httpClient.delete(`/api/knowledge/libraries/${libraryId}`)
      return response
    } catch (error) {
      console.error('删除知识库失败:', error)
      throw error
    }
  }

  /**
   * 添加文档到知识库
   * @param {Object} documentData - 文档数据
   * @param {number} documentData.library_id - 知识库ID
   * @param {string} documentData.title - 文档标题
   * @param {string} documentData.content - 文档内容
   * @returns {Promise} API响应
   */
  async addDocument(documentData) {
    try {
      const response = await httpClient.post('/api/knowledge/documents', documentData)
      return response
    } catch (error) {
      console.error('添加文档失败:', error)
      throw error
    }
  }

  /**
   * 更新文档
   * @param {number} documentId - 文档ID
   * @param {Object} documentData - 文档数据
   * @returns {Promise} API响应
   */
  async updateDocument(documentId, documentData) {
    try {
      const response = await httpClient.put(`/api/knowledge/documents/${documentId}`, documentData)
      return response
    } catch (error) {
      console.error('更新文档失败:', error)
      throw error
    }
  }

  /**
   * 删除文档
   * @param {number} documentId - 文档ID
   * @returns {Promise} API响应
   */
  async deleteDocument(documentId) {
    try {
      const response = await httpClient.delete(`/api/knowledge/documents/${documentId}`)
      return response
    } catch (error) {
      console.error('删除文档失败:', error)
      throw error
    }
  }

  /**
   * 开始解析文档（用户点击解析按钮）
   * @param {number} documentId - 文档ID
   * @returns {Promise} API响应
   */
  async startDocumentProcessing(documentId) {
    try {
      const response = await httpClient.post(`/api/knowledge/documents/${documentId}/process`)
      return response
    } catch (error) {
      console.error('开始解析文档失败:', error)
      throw error
    }
  }

  /**
   * 仅向量化文档（不做图谱）
   * @param {number} documentId - 文档ID
   * @returns {Promise} API响应
   */
  async startDocumentVectorize(documentId) {
    try {
      const response = await httpClient.post(`/api/knowledge/documents/${documentId}/vectorize`)
      return response
    } catch (error) {
      console.error('开始向量化文档失败:', error)
      throw error
    }
  }

  /**
   * 仅图谱化文档（不做向量）
   * @param {number} documentId - 文档ID
   * @returns {Promise} API响应
   */
  async startDocumentGraph(documentId) {
    try {
      const response = await httpClient.post(`/api/knowledge/documents/${documentId}/graph`)
      return response
    } catch (error) {
      console.error('开始图谱化文档失败:', error)
      throw error
    }
  }

  /**
   * 获取处理队列状态
   * @returns {Promise} API响应，包含队列信息和正在处理的文档列表
   */
  async getQueueStatus() {
    try {
      const response = await httpClient.get('/api/knowledge/processing/queue-status')
      return response
    } catch (error) {
      console.error('获取队列状态失败:', error)
      throw error
    }
  }

  /**
   * 获取文档内容（用于预览）
   * @param {number} documentId - 文档ID
   * @returns {Promise} API响应，包含 content / content_type / url 等信息
   */
  async getDocumentContent(documentId) {
    try {
      const response = await httpClient.get(`/api/knowledge/documents/${documentId}/content`)
      return response
    } catch (error) {
      console.error('获取文档内容失败:', error)
      throw error
    }
  }

  /**
   * 获取文档分块信息
   * @param {number} documentId - 文档ID
   * @returns {Promise} API响应，包含 chunks数组和总数
   */
  async getDocumentChunks(documentId) {
    try {
      const response = await httpClient.get(`/api/knowledge/documents/${documentId}/chunks`)
      return response
    } catch (error) {
      console.error('获取文档分块失败:', error)
      throw error
    }
  }

  /**
   * 获取文件上传URL
   * @param {string} documentName - 文档名称
   * @returns {Promise} API响应
   */
  async getUploadUrl(documentName) {
    try {
      const response = await httpClient.post('/api/knowledge/upload-url', {
        document_name: documentName
      })
      return response
    } catch (error) {
      console.error('获取上传URL失败:', error)
      throw error
    }
  }

  /**
   * 上传文件到OSS
   * @param {string} uploadUrl - 上传URL
   * @param {File} file - 文件对象
   * @returns {Promise} 上传响应
   */
  async uploadFileToOSS(uploadUrl, file) {
    try {
      let url = uploadUrl
      let headers = {}

      if (typeof uploadUrl === 'object' && uploadUrl !== null) {
        url = uploadUrl.url
        headers = uploadUrl.signed_headers || {}
      }

      const buffer = await file.arrayBuffer()
      const response = await fetch(url, {
        method: 'PUT',
        body: buffer,
        headers: headers
      })
      
      if (!response.ok) {
        throw new Error(`上传失败: ${response.status} ${response.statusText}`)
      }
      
      const uploadedUrl = (typeof uploadUrl === 'string' ? uploadUrl : uploadUrl.url) || ''
      return {
        success: true,
        url: uploadedUrl.split('?')[0]
      }
    } catch (error) {
      console.error('上传文件到OSS失败:', error)
      throw error
    }
  }

  /**
   * 爬取网站内容
   * @param {Object} crawlData - 爬取数据
   * @param {string} crawlData.url - 网站URL
   * @param {number} crawlData.library_id - 知识库ID
   * @param {number} [crawlData.max_pages] - 最大页面数
   * @returns {Promise} API响应
   */
  async crawlSite(crawlData) {
    try {
      const response = await httpClient.post('/api/crawl/site', crawlData)
      return response
    } catch (error) {
      console.error('爬取网站失败:', error)
      throw error
    }
  }

  /**
   * 获取知识图谱数据
   * @param {string} collectionId - 集合ID
   * @param {string} label - 标签过滤器
   * @returns {Promise} API响应
   */
  async getKnowledgeGraph(collectionId, label = '*') {
    try {
      const response = await httpClient.get(`/api/visual/graph/${collectionId}`, {
        label: label
      })
      return response
    } catch (error) {
      console.error('获取知识图谱失败:', error)
      throw error
    }
  }
}

// 创建并导出API实例
export const knowledgeAPI = new KnowledgeLibraryAPI()

// 导出默认实例
export default knowledgeAPI
