/**
 * 聊天相关API接口
 * 对接后端 /workspace/rag-zxj/rag-demo/backend/api/chat.py
 */

import { httpClient } from './config.js'

/**
 * 发送聊天消息（非流式）
 * @param {Object} chatData - 聊天数据
 * @param {string} chatData.content - 消息内容
 * @param {string} [chatData.conversation_id] - 会话ID（可选）
 * @param {string} [chatData.user_id] - 用户ID（可选）
 * @returns {Promise<Object>} 聊天响应
 */
export async function sendMessage(chatData) {
  try {
    const response = await httpClient.post('/llm/chat', chatData)
    return response
  } catch (error) {
    console.error('发送消息失败:', error)
    throw new Error(error.message || '发送消息失败')
  }
}

/**
 * 发送聊天消息（流式响应）
 * @param {Object} chatData - 聊天数据
 * @param {Function} onMessage - 消息回调函数
 * @param {Function} [onError] - 错误回调函数
 * @param {Function} [onComplete] - 完成回调函数
 * @returns {Promise<void>}
 */
export async function sendMessageStream(chatData, onMessage, onError, onComplete) {
  try {
    const url = `/llm/chat/stream`
    console.log('🚀 开始流式请求:', url, chatData)
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...httpClient.getHeaders()
      },
      body: JSON.stringify(chatData)
    })

    console.log('📡 收到响应:', response.status, response.statusText)
    console.log('📋 响应头:', Object.fromEntries(response.headers.entries()))

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    console.log('🔄 开始读取流式数据...')

    try {
      let chunkCount = 0
      let buffer = ''
      const processEventBlock = (part) => {
        if (!part.trim()) return

        const lines = part.split('\n').filter(Boolean)
        const dataLines = lines
          .filter(line => line.startsWith('data:'))
          .map(line => line.replace(/^data:\s?/, ''))

        if (!dataLines.length) {
          console.log('🔸 非data事件块:', part)
          return
        }

        const jsonStr = dataLines.join('\n')
        try {
          console.log('🔍 解析JSON字符串:', jsonStr)
          const data = JSON.parse(jsonStr)
          console.log('✨ 解析成功的数据:', data)

          if (data.type === 'error') {
            console.error('❌ 收到错误数据:', data)
            onError && onError(new Error(data.message || data.error))
          } else {
            console.log('📨 调用onMessage回调，数据类型:', data.type)
            onMessage && onMessage(data)
          }
        } catch (parseError) {
          console.warn('⚠️ 解析流式数据失败:', parseError, '原始事件块:', part)
        }
      }

      while (true) {
        const { done, value } = await reader.read()
        
        if (done) {
          console.log('✅ 流式数据读取完成，总共处理了', chunkCount, '个数据块')
          const remaining = buffer.replace(/\r\n|\r/g, '\n')
          if (remaining.trim()) {
            processEventBlock(remaining)
          }
          onComplete && onComplete()
          break
        }

        chunkCount++
        const chunk = decoder.decode(value, { stream: true })
        console.log(`📦 收到第${chunkCount}个数据块:`, chunk)

        buffer += chunk
        const normalized = buffer.replace(/\r\n|\r/g, '\n')
        const parts = normalized.split('\n\n')
        buffer = parts.pop() || ''

        for (const part of parts) {
          processEventBlock(part)
        }
      }
    } finally {
      reader.releaseLock()
      console.log('🔓 释放reader锁')
    }
  } catch (error) {
    console.error('💥 流式聊天请求失败:', error)
    onError && onError(error)
  }
}

/**
 * 获取聊天历史记录
 * @param {string} userId - 用户ID
 * @param {string} [conversationId] - 会话ID（可选）
 * @returns {Promise<Object>} 聊天历史响应
 */
export async function getChatHistory(userId, conversationId = null) {
  try {
    const params = conversationId ? { conversation_id: conversationId } : {}
    const response = await httpClient.get(`/llm/history/${userId}`, params)
    return response
  } catch (error) {
    console.error('获取聊天历史失败:', error)
    throw new Error(error.message || '获取聊天历史失败')
  }
}

/**
 * 获取单轮对话历史记录
 * @param {string} conversationId - 会话ID
 * @returns {Promise<Object>} 单轮对话历史响应
 */
export async function getSingleConversationHistory(conversationId) {
  try {
    const response = await httpClient.get(`/llm/history/single/${conversationId}`)
    return response
  } catch (error) {
    console.error('获取单轮对话历史失败:', error)
    throw new Error(error.message || '获取单轮对话历史失败')
  }
}

/**
 * 获取所有历史记录标题列表
 * @param {string} userId - 用户ID
 * @returns {Promise<Object>} 历史记录标题列表响应
 */
export async function getChatHistoryTitles(userId) {
  try {
    const response = await httpClient.get(`/llm/history/titles/${userId}`)
    return response
  } catch (error) {
    console.error('获取历史记录标题列表失败:', error)
    throw new Error(error.message || '获取历史记录标题列表失败')
  }
}

/**
 * 获取签名URL
 * @returns {Promise<Object>} 签名URL响应
 */
export async function getSignatureUrl() {
  try {
    const response = await httpClient.post('/llm/get-url')
    return response
  } catch (error) {
    console.error('获取签名URL失败:', error)
    throw new Error(error.message || '获取签名URL失败')
  }
}

/**
 * 添加聊天历史记录
 * @param {string} userId - 用户ID
 * @param {string} conversationId - 会话ID
 * @param {Array} messages - 消息列表
 * @returns {Promise<Object>} 响应结果
 */
export async function addChatHistory(userId, conversationId, messages) {
  try {
    const response = await httpClient.post('/llm/history', {
      user_id: userId,
      conversation_id: conversationId,
      messages: messages
    })
    return response
  } catch (error) {
    console.error('添加聊天历史记录失败:', error)
    throw new Error(error.message || '添加聊天历史记录失败')
  }
}

/**
 * 创建新对话
 * @param {string} userId - 用户ID
 * @param {string} [title] - 对话标题（可选）
 * @returns {Promise<Object>} 创建结果
 */
export async function createConversation(userId, title = null) {
  try {
    const response = await httpClient.post('/llm/conversation', {
      user_id: userId,
      title: title
    })
    return response
  } catch (error) {
    console.error('创建对话失败:', error)
    throw new Error(error.message || '创建对话失败')
  }
}

/**
 * 删除对话
 * @param {string} conversationId - 对话ID
 * @returns {Promise<Object>} 删除结果
 */
export async function deleteConversation(conversationId) {
  try {
    const response = await httpClient.delete(`/llm/conversation/${conversationId}`)
    return response
  } catch (error) {
    console.error('删除对话失败:', error)
    throw new Error(error.message || '删除对话失败')
  }
}

export async function getUserMemory(userId, collectionId, profileLimit = 50, eventLimit = 50, conversationId = null, shortTermLimit = 10) {
  try {
    const params = new URLSearchParams({
      collection_id: collectionId,
      profile_limit: String(profileLimit),
      event_limit: String(eventLimit)
    })
    if (conversationId) {
      params.set('conversation_id', String(conversationId))
      params.set('short_term_limit', String(shortTermLimit))
    }
    const response = await httpClient.get(`/llm/memory/${userId}?${params.toString()}`)
    return response
  } catch (error) {
    console.error('获取用户记忆失败:', error)
    throw new Error(error.message || '获取用户记忆失败')
  }
}

export async function deleteUserMemoryProfile(userId, memoryKey, collectionId) {
  try {
    const params = new URLSearchParams({ collection_id: collectionId })
    const response = await httpClient.delete(`/llm/memory/${userId}/profile/${encodeURIComponent(memoryKey)}?${params.toString()}`)
    return response
  } catch (error) {
    console.error('删除画像记忆失败:', error)
    throw new Error(error.message || '删除画像记忆失败')
  }
}

export async function deleteConversationMemory(userId, conversationId, collectionId) {
  try {
    const params = new URLSearchParams({ collection_id: collectionId })
    const response = await httpClient.delete(`/llm/memory/${userId}/conversation/${conversationId}?${params.toString()}`)
    return response
  } catch (error) {
    console.error('删除会话记忆失败:', error)
    throw new Error(error.message || '删除会话记忆失败')
  }
}

export async function deleteAllUserMemory(userId, collectionId) {
  try {
    const params = new URLSearchParams({ collection_id: collectionId })
    const response = await httpClient.delete(`/llm/memory/${userId}?${params.toString()}`)
    return response
  } catch (error) {
    console.error('删除全部记忆失败:', error)
    throw new Error(error.message || '删除全部记忆失败')
  }
}

/**
 * 流式聊天辅助类
 * 用于管理流式聊天的状态和事件
 */
export class StreamChatManager {
  constructor() {
    this.isStreaming = false
    this.currentMessage = ''
    this.onMessageCallbacks = []
    this.onErrorCallbacks = []
    this.onCompleteCallbacks = []
  }

  /**
   * 添加消息回调
   * @param {Function} callback - 回调函数
   */
  onMessage(callback) {
    this.onMessageCallbacks.push(callback)
  }

  /**
   * 添加错误回调
   * @param {Function} callback - 回调函数
   */
  onError(callback) {
    this.onErrorCallbacks.push(callback)
  }

  /**
   * 添加完成回调
   * @param {Function} callback - 回调函数
   */
  onComplete(callback) {
    this.onCompleteCallbacks.push(callback)
  }

  /**
   * 开始流式聊天
   * @param {Object} chatData - 聊天数据
   */
  async startStream(chatData) {
    if (this.isStreaming) {
      throw new Error('已有流式聊天正在进行中')
    }

    this.isStreaming = true
    this.currentMessage = ''

    try {
      await sendMessageStream(
        chatData,
        (data) => {
          this.currentMessage += data.content || ''
          this.onMessageCallbacks.forEach(callback => callback(data, this.currentMessage))
        },
        (error) => {
          this.isStreaming = false
          this.onErrorCallbacks.forEach(callback => callback(error))
        },
        () => {
          this.isStreaming = false
          this.onCompleteCallbacks.forEach(callback => callback(this.currentMessage))
        }
      )
    } catch (error) {
      this.isStreaming = false
      throw error
    }
  }

  /**
   * 停止流式聊天
   */
  stopStream() {
    this.isStreaming = false
  }

  /**
   * 清理回调函数
   */
  clearCallbacks() {
    this.onMessageCallbacks = []
    this.onErrorCallbacks = []
    this.onCompleteCallbacks = []
  }
}
