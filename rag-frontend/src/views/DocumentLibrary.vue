<template>
  <div class="h-screen flex bg-amber-50/30">
    <!-- 左侧边栏 -->
    <div class="w-80 bg-white border-r border-gray-100 flex flex-col shadow-sm">
      <!-- 头部 -->
      <div class="p-4 border-b border-gray-100">
        <div class="flex items-center justify-between mb-4">
          <h1 class="text-xl font-normal text-gray-900">文档库管理</h1>
          <router-link
            to="/chat"
            class="text-gray-500 hover:text-gray-700 p-2 rounded-lg hover:bg-gray-50"
          >
            <svg
              class="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M10 19l-7-7m0 0l7-7m-7 7h18"
              />
            </svg>
          </router-link>
        </div>
        <button
          @click="showCreateDialog = true"
          :disabled="loading"
          class="w-full bg-gray-900 text-white py-2 px-4 rounded-lg hover:bg-gray-800 transition-colors flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <svg
            class="w-4 h-4 mr-2"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M12 4v16m8-8H4"
            />
          </svg>
          新建文档库
        </button>
      </div>

      <!-- 文档库列表 -->
      <div class="flex-1 overflow-y-auto p-4">
        <!-- 加载状态 -->
        <div
          v-if="loading && documentLibraries.length === 0"
          class="flex items-center justify-center py-8"
        >
          <div
            class="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"
          ></div>
          <span class="ml-2 text-gray-600 font-light">加载中...</span>
        </div>

        <!-- 错误状态 -->
        <div
          v-else-if="error && documentLibraries.length === 0"
          class="text-center py-8"
        >
          <div class="text-red-500 mb-2 font-light">{{ error }}</div>
          <button
            @click="loadLibraries"
            class="text-gray-900 hover:text-gray-700 text-sm font-normal"
          >
            重新加载
          </button>
        </div>

        <!-- 空状态 -->
        <div
          v-else-if="documentLibraries.length === 0"
          class="text-center py-8 text-gray-500"
        >
          <svg
            class="w-12 h-12 mx-auto mb-4 text-gray-300"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          <p class="font-normal">暂无文档库</p>
          <p class="text-sm font-light">点击上方按钮创建第一个文档库</p>
        </div>

        <!-- 文档库列表 -->
        <div v-else class="space-y-2">
          <div
            v-for="library in documentLibraries"
            :key="library.id"
            @click="selectLibrary(library)"
            class="p-3 rounded-lg cursor-pointer transition-colors group border"
            :class="{
              'bg-amber-50 border-amber-200':
                selectedLibrary && selectedLibrary.id === library.id,
              'hover:bg-gray-50 border-gray-100':
                !selectedLibrary || selectedLibrary.id !== library.id,
            }"
          >
            <div class="flex items-start justify-between">
              <div class="flex-1 min-w-0">
                <h3 class="text-sm font-normal text-gray-900 truncate">
                  {{ library.title }}
                </h3>
                <p class="text-xs text-gray-500 mt-1 line-clamp-2 font-light">
                  {{ library.description || "暂无描述" }}
                </p>
                <div
                  class="flex items-center mt-2 text-xs text-gray-400 font-light"
                >
                  <span>{{ (library.documents && library.documents.length) || 0 }} 个文档</span>
                  <span class="mx-1">•</span>
                  <span>{{ formatTime(library.updated_at) }}</span>
                </div>
              </div>
              <div
                class="flex space-x-1 opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <button
                  @click.stop="editLibrary(library)"
                  :disabled="loading"
                  class="text-gray-400 hover:text-gray-900 p-1 rounded transition-colors disabled:opacity-50"
                >
                  <svg
                    class="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                    />
                  </svg>
                </button>
                <button
                  @click.stop="deleteLibrary(library.id)"
                  :disabled="loading"
                  class="text-gray-400 hover:text-red-500 p-1 rounded transition-colors disabled:opacity-50"
                >
                  <svg
                    class="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                    />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 主内容区域 -->
    <div class="flex-1 flex flex-col">
      <!-- 文档库详情头部 -->
      <div
        class="bg-white/80 backdrop-blur-sm border-b border-gray-100 p-4 shadow-sm"
        v-if="selectedLibrary"
      >
        <div class="flex items-center justify-between">
          <div>
            <h2 class="text-lg font-normal text-gray-900">
              {{ selectedLibrary.title }}
            </h2>
            <p class="text-sm text-gray-600 mt-1 font-light">
              {{ selectedLibrary.description || "暂无描述" }}
            </p>
          </div>
          <div class="flex space-x-3">
            <!-- 队列状态按钮 -->
            <button
              @click="showQueueDrawer = true"
              :disabled="loading"
              class="bg-gray-100 text-gray-700 px-3 py-2 rounded-lg hover:bg-gray-200 transition-colors flex items-center text-sm relative disabled:opacity-50 disabled:cursor-not-allowed"
              title="查看处理队列"
            >
              <svg class="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
              处理队列
              <!-- 径标提示 -->
              <span v-if="queueStatus && (queueStatus.processing_count > 0 || queueStatus.queue_size > 0)"
                    class="absolute -top-1 -right-1 w-5 h-5 bg-blue-500 text-white text-xs rounded-full flex items-center justify-center">
                {{ queueStatus.processing_count + queueStatus.queue_size }}
              </span>
            </button>
            
            <button
              @click="batchVectorize"
              :disabled="loading || !hasUnvectorizedDocs"
              class="bg-gray-700 text-white px-3 py-2 rounded-lg hover:bg-gray-600 transition-colors flex items-center text-sm disabled:opacity-50 disabled:cursor-not-allowed"
              title="将未向量化的文档一次性加入队列"
            >
              <svg class="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              一键向量化
            </button>
            <button
              @click="batchGraph"
              :disabled="loading || !hasUngraphedDocs"
              class="bg-blue-600 text-white px-3 py-2 rounded-lg hover:bg-blue-700 transition-colors flex items-center text-sm disabled:opacity-50 disabled:cursor-not-allowed"
              title="将未图谱化的文档一次性加入队列"
            >
              <svg class="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
              </svg>
              一键图谱化
            </button>
            <button
              @click="viewKnowledgeGraph"
              :disabled="loading"
              class="bg-gray-800 text-white px-4 py-2 rounded-lg hover:bg-gray-700 transition-colors flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <svg
                class="w-4 h-4 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"
                />
              </svg>
              知识图谱
            </button>
            <button
              @click="showAddDocumentDialog = true"
              :disabled="loading"
              class="bg-gray-900 text-white px-4 py-2 rounded-lg hover:bg-gray-800 transition-colors flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <svg
                class="w-4 h-4 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M12 4v16m8-8H4"
                />
              </svg>
              添加文档
            </button>
          </div>
        </div>
      </div>

      <!-- 文档列表 -->
      <div class="flex-1 overflow-y-auto p-4" v-if="selectedLibrary">
        <!-- 文档加载状态 -->
        <div
          v-if="documentLoading"
          class="flex items-center justify-center py-8"
        >
          <div
            class="animate-spin rounded-full h-6 w-6 border-b-2 border-gray-900"
          ></div>
          <span class="ml-2 text-gray-600 font-light">加载文档中...</span>
        </div>

        <!-- 文档列表 -->
        <div
          v-else-if="
            selectedLibrary.documents && selectedLibrary.documents.length > 0
          "
          class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
        >
          <div
            v-for="document in selectedLibrary.documents"
            :key="document.id"
            class="bg-white rounded-lg border border-gray-100 p-4 hover:shadow-sm transition-shadow group"
          >
            <div class="flex items-start justify-between mb-3">
              <div class="flex items-center">
                <div
                  class="w-8 h-8 rounded-lg flex items-center justify-center mr-3"
                  :class="getDocumentTypeClass(document.type)"
                >
                  <svg
                    v-if="document.type === 'link'"
                    class="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.1m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"
                    />
                  </svg>
                  <svg
                    v-else
                    class="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                </div>
                <div class="flex-1 min-w-0">
                  <h4 class="text-sm font-normal text-gray-900 truncate">
                    {{ document.name }}
                  </h4>
                  <p class="text-xs text-gray-500 font-light">
                    {{ document.type === "link" ? "网站链接" : "文件" }}
                  </p>
                </div>
              </div>
              <div class="flex space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  v-if="document.type === 'file' && document.url && (document.name.endsWith('.md') || document.name.endsWith('.txt'))"
                  @click="previewDocument(document)"
                  :disabled="loading"
                  class="text-gray-400 hover:text-gray-900 p-1 rounded transition-all disabled:opacity-50"
                  title="预览文档"
                >
                  <svg
                    class="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                    />
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                    />
                  </svg>
                </button>
                <button
                  v-if="document.is_vectorized"
                  @click="viewDocumentChunks(document)"
                  :disabled="loading"
                  class="text-gray-400 hover:text-blue-600 p-1 rounded transition-all disabled:opacity-50"
                  title="查看分块"
                >
                  <svg
                    class="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M4 6h16M4 10h16M4 14h16M4 18h16"
                    />
                  </svg>
                </button>
                <button
                  @click="removeDocument(document.id)"
                  :disabled="loading"
                  class="text-gray-400 hover:text-red-500 p-1 rounded transition-all disabled:opacity-50"
                  title="删除文档"
                >
                  <svg
                    class="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                    />
                  </svg>
                </button>
              </div>
            </div>

            <div v-if="document.url" class="text-xs text-gray-700 truncate mb-2">
              <a :href="document.url" target="_blank" class="hover:underline">
                {{ document.url }}
              </a>
            </div>

            <!-- 解析状态和按钮 -->
            <div class="mt-3 pt-3 border-t border-gray-100">
              <!-- 状态显示 -->
              <div class="flex items-center gap-3 text-xs font-light mb-3">
                <span v-if="document.is_vectorized" class="text-green-600 flex items-center">
                  <svg class="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
                  </svg>
                  已向量化
                </span>
                <span v-else class="text-gray-400 flex items-center">
                  <svg class="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 000 2h6a1 1 0 100-2H7z" clip-rule="evenodd" />
                  </svg>
                  未向量化
                </span>
                
                <span v-if="document.is_graphed" class="text-blue-600 flex items-center">
                  <svg class="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
                  </svg>
                  已图谱化
                </span>
                <span v-else class="text-gray-400 flex items-center">
                  <svg class="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 000 2h6a1 1 0 100-2H7z" clip-rule="evenodd" />
                  </svg>
                  未图谱化
                </span>
              </div>
              
              <!-- 解析按钮（仅文件类型显示） -->
              <div v-if="document.type === 'file'" class="flex gap-2">
                <button
                  v-if="!document.is_vectorized"
                  @click="startVectorize(document.id)"
                  :disabled="loading || processingDocuments.has(document.id)"
                  class="flex-1 text-xs px-3 py-1.5 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  <svg
                    v-if="processingDocuments.has(document.id)"
                    class="animate-spin h-3 w-3 mr-1.5"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  {{ processingDocuments.has(document.id) ? '向量化中...' : '向量解析' }}
                </button>
                
                <button
                  v-if="!document.is_graphed"
                  @click="startGraph(document.id)"
                  :disabled="loading || graphingDocuments.has(document.id)"
                  class="flex-1 text-xs px-3 py-1.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  <svg
                    v-if="graphingDocuments.has(document.id)"
                    class="animate-spin h-3 w-3 mr-1.5"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  {{ graphingDocuments.has(document.id) ? '图谱化中...' : '图谱解析' }}
                </button>
                
                <span v-if="document.is_vectorized && document.is_graphed" class="flex-1 text-xs text-gray-400 flex items-center justify-center">
                  {{ formatTime(document.created_at) }}
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- 空文档状态 -->
        <div v-else class="text-center py-12 text-gray-500">
          <svg
            class="w-16 h-16 mx-auto mb-4 text-gray-300"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          <p class="text-lg mb-2 font-normal">暂无文档</p>
          <p class="text-sm font-light">点击右上角按钮添加第一个文档</p>
        </div>
      </div>

      <!-- 未选择文档库状态 -->
      <div v-else class="flex-1 flex items-center justify-center text-gray-500">
        <div class="text-center">
          <svg
            class="w-20 h-20 mx-auto mb-4 text-gray-300"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
            />
          </svg>
          <p class="text-lg mb-2 font-normal">选择一个文档库</p>
          <p class="text-sm font-light">
            从左侧列表中选择文档库来查看和管理文档
          </p>
        </div>
      </div>
    </div>

    <!-- 创建/编辑文档库对话框 -->
    <div
      v-if="showCreateDialog || showEditDialog"
      class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
    >
      <div
        class="bg-white rounded-xl shadow-[0_4px_16px_rgba(0,0,0,0.12)] border border-gray-100 p-6 w-full max-w-md mx-4"
      >
        <h3 class="text-lg font-normal text-gray-900 mb-4">
          {{ showEditDialog ? "编辑文档库" : "创建文档库" }}
        </h3>
        <form
          @submit.prevent="showEditDialog ? updateLibrary() : createLibrary()"
        >
          <div class="mb-4">
            <label class="block text-sm font-normal text-gray-700 mb-2"
              >标题</label
            >
            <input
              v-model="libraryForm.title"
              type="text"
              required
              :disabled="loading"
              class="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent disabled:opacity-50 bg-white"
              placeholder="请输入文档库标题"
            />
          </div>
          <div class="mb-6">
            <label class="block text-sm font-normal text-gray-700 mb-2"
              >描述</label
            >
            <textarea
              v-model="libraryForm.description"
              rows="3"
              :disabled="loading"
              class="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent disabled:opacity-50 bg-white"
              placeholder="请输入文档库描述（可选）"
            ></textarea>
          </div>
          <div v-if="!showEditDialog" class="mb-6">
            <label class="block text-sm font-normal text-gray-700 mb-2"
              >校验码</label
            >
            <input
              v-model="libraryForm.verificationCode"
              type="text"
              required
              :disabled="loading"
              class="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent disabled:opacity-50 bg-white"
              placeholder="请输入校验码"
            />
          </div>
          <div class="flex justify-end space-x-3">
            <button
              type="button"
              @click="closeDialog"
              :disabled="loading"
              class="px-4 py-2 text-gray-700 bg-gray-50 border border-gray-200 rounded-lg hover:bg-gray-100 hover:border-gray-300 transition-colors disabled:opacity-50"
            >
              取消
            </button>
            <button
              type="submit"
              :disabled="loading"
              class="px-4 py-2 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition-colors disabled:opacity-50 flex items-center"
            >
              <div
                v-if="loading"
                class="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"
              ></div>
              {{ showEditDialog ? "更新" : "创建" }}
            </button>
          </div>
        </form>
      </div>
    </div>

    <!-- 添加文档对话框 -->
    <div
      v-if="showAddDocumentDialog"
      class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
    >
      <div
        class="bg-white rounded-xl shadow-[0_4px_16px_rgba(0,0,0,0.12)] border border-gray-100 p-6 w-full max-w-md mx-4"
      >
        <h3 class="text-lg font-normal text-gray-900 mb-4">添加文档</h3>
        <form @submit.prevent="addDocument">
          <div class="mb-4">
            <label class="block text-sm font-normal text-gray-700 mb-2"
              >文档类型</label
            >
            <select
              v-model="documentForm.type"
              :disabled="loading"
              class="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent disabled:opacity-50 bg-white"
            >
              <option value="link">网站链接</option>
              <option value="file">文件上传</option>
            </select>
          </div>

          <div v-if="documentForm.type === 'link'" class="mb-4">
            <label class="block text-sm font-normal text-gray-700 mb-2"
              >链接地址</label
            >
            <input
              v-model="documentForm.url"
              type="url"
              required
              :disabled="loading"
              class="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent disabled:opacity-50 bg-white"
              placeholder="https://example.com"
            />
          </div>

          <div v-if="documentForm.type === 'link'" class="mb-4">
            <label class="block text-sm font-normal text-gray-700 mb-2"
              >URL前缀限制</label
            >
            <input
              v-model="documentForm.prefix"
              type="text"
              :disabled="loading"
              class="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent disabled:opacity-50 bg-white"
              placeholder="留空则自动使用URL根路径"
            />
            <p class="text-xs text-gray-500 mt-1 font-light">
              只爬取以此前缀开头的URL，留空则自动使用输入URL的根路径
            </p>
          </div>

          <div v-if="documentForm.type === 'file'" class="mb-4">
            <label class="block text-sm font-normal text-gray-700 mb-2"
              >选择文件</label
            >
            <input
              @change="handleFileSelect"
              type="file"
              accept=".pdf,.doc,.docx,.md,.txt"
              required
              :disabled="loading"
              class="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent disabled:opacity-50 bg-white"
            />
            <p class="text-xs text-gray-500 mt-1 font-light">
              支持 PDF、DOC、DOCX、MD、TXT 格式
            </p>
          </div>

          <div class="mb-6">
            <label class="block text-sm font-normal text-gray-700 mb-2"
              >文档名称</label
            >
            <input
              v-model="documentForm.name"
              type="text"
              required
              :disabled="loading"
              class="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent disabled:opacity-50 bg-white"
              placeholder="请输入文档名称"
            />
          </div>

          <div class="flex justify-end space-x-3">
            <button
              type="button"
              @click="showAddDocumentDialog = false"
              :disabled="loading"
              class="px-4 py-2 text-gray-700 bg-gray-50 border border-gray-200 rounded-lg hover:bg-gray-100 hover:border-gray-300 transition-colors disabled:opacity-50"
            >
              取消
            </button>
            <button
              type="submit"
              :disabled="loading"
              class="px-4 py-2 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition-colors disabled:opacity-50 flex items-center"
            >
              <div
                v-if="loading"
                class="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"
              ></div>
              添加
            </button>
          </div>
        </form>
      </div>
    </div>
    
    <!-- 文档预览弹窗 -->
    <div
      v-if="showPreviewDialog"
      class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
    >
      <div
        class="bg-white rounded-xl shadow-xl border border-gray-100 w-full max-w-4xl max-h-[90vh] flex flex-col"
      >
        <!-- 预览头部 -->
        <div class="flex items-center justify-between p-4 border-b border-gray-200">
          <h3 class="text-lg font-normal text-gray-900">{{ previewData.name }}</h3>
          <button
            @click="showPreviewDialog = false"
            class="text-gray-400 hover:text-gray-600 p-1 rounded hover:bg-gray-100 transition-colors"
          >
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        <!-- 预览内容区域 -->
        <div class="flex-1 overflow-y-auto p-6">
          <!-- 加载中 -->
          <div v-if="previewLoading" class="flex items-center justify-center py-12">
            <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
            <span class="ml-2 text-gray-600">加载中...</span>
          </div>

          <!-- 预览内容（文本） -->
          <div
            v-else-if="previewData.content"
            class="prose prose-sm max-w-none"
            v-html="renderPreviewMarkdown(previewData.content)"
          ></div>

          <!-- 无内容或不支持 -->
          <div v-else class="text-center py-12 text-gray-500">
            <svg
              class="w-16 h-16 mx-auto mb-4 text-gray-300"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
            <p class="text-lg mb-2">暂无内容预览</p>
            <p class="text-sm">该文档类型当前不支持在线预览</p>
            <a
              v-if="previewData.url"
              :href="previewData.url"
              target="_blank"
              class="inline-block mt-4 text-gray-900 hover:underline"
            >
              点击查看原文件
            </a>
          </div>
        </div>
      </div>
    </div>
    
    <!-- 文档分块查看对话框 -->
    <div
      v-if="showChunksDialog"
      class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
    >
      <div
        class="bg-white rounded-xl shadow-xl border border-gray-100 w-full max-w-5xl max-h-[90vh] flex flex-col"
      >
        <!-- 头部 -->
        <div class="flex items-center justify-between p-4 border-b border-gray-200">
          <div>
            <h3 class="text-lg font-normal text-gray-900">{{ chunksData.document_name }}</h3>
            <p class="text-sm text-gray-500 mt-1">共 {{ chunksData.total_chunks }} 个分块</p>
          </div>
          <button
            @click="showChunksDialog = false"
            class="text-gray-400 hover:text-gray-600 p-1 rounded hover:bg-gray-100 transition-colors"
          >
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        <!-- 内容区域 -->
        <div class="flex-1 overflow-y-auto p-6">
          <!-- 加载中 -->
          <div v-if="chunksLoading" class="flex items-center justify-center py-12">
            <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
            <span class="ml-2 text-gray-600">加载中...</span>
          </div>

          <!-- 无分块 -->
          <div v-else-if="chunksData.total_chunks === 0" class="text-center py-12 text-gray-500">
            <svg
              class="w-16 h-16 mx-auto mb-4 text-gray-300"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
            <p class="text-lg mb-2">未找到分块</p>
            <p class="text-sm">{{ chunksData.message || '该文档未进行向量化或分块失败' }}</p>
          </div>

          <!-- 分块列表 -->
          <div v-else class="space-y-4">
            <div
              v-for="chunk in chunksData.chunks"
              :key="chunk.chunk_index"
              class="border border-gray-200 rounded-lg p-4 hover:border-blue-300 transition-colors"
            >
              <!-- 分块头部 -->
              <div class="flex items-center justify-between mb-3">
                <div class="flex items-center gap-3">
                  <span class="inline-flex items-center justify-center w-8 h-8 rounded-full bg-blue-100 text-blue-700 text-sm font-semibold">
                    {{ chunk.chunk_index }}
                  </span>
                  <div class="text-sm text-gray-600">
                    <span>长度: {{ chunk.chunk_size }} 字符</span>
                  </div>
                </div>
              </div>
              
              <!-- 分块内容 -->
              <div class="bg-gray-50 rounded-lg p-3 mt-2">
                <pre class="text-xs text-gray-700 whitespace-pre-wrap font-mono leading-relaxed">{{ chunk.text }}</pre>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- 右侧队列抽屉 -->
    <div v-if="showQueueDrawer" 
         class="fixed inset-0 bg-black bg-opacity-30 z-50 flex justify-end"
         @click.self="showQueueDrawer = false">
      <div class="w-96 bg-white h-full shadow-2xl overflow-hidden flex flex-col animate-slide-in-right">
        <!-- 抽屉头部 -->
        <div class="bg-gradient-to-r from-blue-500 to-indigo-600 text-white p-4 flex items-center justify-between">
          <div class="flex items-center">
            <div class="animate-pulse w-2 h-2 bg-white rounded-full mr-3"></div>
            <h3 class="text-lg font-medium">处理队列</h3>
          </div>
          <button @click="showQueueDrawer = false" 
                  class="text-white hover:bg-white/20 p-1.5 rounded-lg transition-colors">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        <!-- 统计信息 -->
        <div class="bg-blue-50 p-4 border-b border-blue-100">
          <div class="flex items-center justify-between text-sm">
            <div class="flex items-center space-x-4">
              <div>
                <span class="text-gray-600">正在处理:</span>
                <span class="ml-2 font-semibold text-blue-600">{{ (queueStatus && queueStatus.processing_count) || 0 }}</span>
              </div>
              <div>
                <span class="text-gray-600">排队中:</span>
                <span class="ml-2 font-semibold text-indigo-600">{{ (queueStatus && queueStatus.queue_size) || 0 }}</span>
              </div>
            </div>
          </div>
        </div>
        
        <!-- 队列内容 -->
        <div class="flex-1 overflow-y-auto p-4">
          <!-- 空状态 -->
          <div v-if="!queueStatus || (queueStatus.processing_count === 0 && queueStatus.queue_size === 0)"
               class="flex flex-col items-center justify-center h-full text-gray-400">
            <svg class="w-20 h-20 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
            <p class="text-lg font-medium">队列为空</p>
            <p class="text-sm mt-1">当前没有文档在处理</p>
          </div>
          
          <!-- 有内容 -->
          <div v-else class="space-y-4">
            <!-- 正在处理的文档 -->
            <div v-if="queueStatus.processing_documents && queueStatus.processing_documents.length > 0">
              <div class="flex items-center mb-3">
                <div class="w-1 h-4 bg-blue-500 rounded-full mr-2"></div>
                <h4 class="text-sm font-semibold text-gray-800">正在处理</h4>
              </div>
              <div class="space-y-2">
                <div v-for="doc in queueStatus.processing_documents" :key="'proc-' + doc.document_id"
                     class="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-3 border-l-4 border-blue-500">
                  <div class="flex items-start justify-between">
                    <div class="flex items-start flex-1 min-w-0 mr-3">
                      <div class="animate-spin rounded-full h-5 w-5 border-2 border-blue-500 border-t-transparent mr-3 mt-0.5 flex-shrink-0"></div>
                      <div class="flex-1 min-w-0">
                        <p class="text-sm font-medium text-gray-900 truncate">{{ doc.document_name }}</p>
                        <div class="flex items-center mt-1 text-xs text-gray-600">
                          <span class="px-2 py-0.5 rounded-full" 
                                :class="doc.mode === 'vectorize' ? 'bg-gray-100 text-gray-700' : (doc.mode === 'graph' ? 'bg-blue-100 text-blue-700' : 'bg-purple-100 text-purple-700')">
                            {{ doc.mode === 'vectorize' ? '向量化' : (doc.mode === 'graph' ? '图谱化' : '完整处理') }}
                          </span>
                          <span class="ml-2">· {{ doc.elapsed_seconds }}s</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <!-- 排队中的文档 -->
            <div v-if="queueStatus.queued_documents && queueStatus.queued_documents.length > 0">
              <div class="flex items-center mb-3">
                <div class="w-1 h-4 bg-indigo-400 rounded-full mr-2"></div>
                <h4 class="text-sm font-semibold text-gray-800">排队中</h4>
              </div>
              <div class="space-y-2">
                <div v-for="doc in queueStatus.queued_documents" :key="'queued-' + doc.document_id"
                     class="bg-gray-50 rounded-lg p-3 border border-gray-200 hover:border-indigo-300 transition-colors">
                  <div class="flex items-start">
                    <div class="w-7 h-7 rounded-full bg-indigo-100 flex items-center justify-center mr-3 flex-shrink-0">
                      <span class="text-xs font-bold text-indigo-600">#{{ doc.queue_position }}</span>
                    </div>
                    <div class="flex-1 min-w-0">
                      <p class="text-sm font-medium text-gray-800 truncate">{{ doc.document_name }}</p>
                      <div class="flex items-center mt-1 text-xs text-gray-500">
                        <span class="px-2 py-0.5 rounded-full bg-gray-100">
                          {{ doc.mode === 'vectorize' ? '待向量化' : (doc.mode === 'graph' ? '待图谱化' : '待处理') }}
                        </span>
                        <span class="ml-2">· 已等待 {{ doc.wait_seconds }}s</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- 抽屉底部 -->
        <div class="border-t border-gray-200 p-4 bg-gray-50">
          <button @click="showQueueDrawer = false" 
                  class="w-full py-2 px-4 bg-gray-800 text-white rounded-lg hover:bg-gray-700 transition-colors">
            关闭
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, onUnmounted, computed } from "vue";
import { useRouter } from "vue-router";
import { ElMessageBox, ElMessage } from "element-plus";
import { knowledgeAPI } from "@/api/knowledge.js";
import { marked } from "marked";

const FRONTEND_VERIFY_CODE = import.meta.env.VITE_KB_VERIFY_CODE || "123456";

// 路由
const router = useRouter();

// 响应式数据
const documentLibraries = ref([]);
const selectedLibrary = ref(null);
const showCreateDialog = ref(false);
const showEditDialog = ref(false);
const showAddDocumentDialog = ref(false);
const showQueueDrawer = ref(false);  // 队列抽屉显示状态
const loading = ref(false);
const documentLoading = ref(false);
const error = ref("");

// 处理状态跟踪：使用 document_id 作为 key
const processingDocuments = ref(new Set()); // 向量化处理中的文档 ID
const graphingDocuments = ref(new Set());   // 图谱化处理中的文档 ID

// 队列状态
const queueStatus = ref(null);
let queueStatusInterval = null;

// 计算属性：是否有未处理的文档
const hasUnvectorizedDocs = computed(() => {
  if (!selectedLibrary.value || !selectedLibrary.value.documents) return false;
  return selectedLibrary.value.documents.some(doc => 
    doc.type === 'file' && !doc.is_vectorized
  );
});

const hasUngraphedDocs = computed(() => {
  if (!selectedLibrary.value || !selectedLibrary.value.documents) return false;
  return selectedLibrary.value.documents.some(doc => 
    doc.type === 'file' && !doc.is_graphed
  );
});

// 表单数据
const libraryForm = reactive({
  title: "",
  description: "",
  verificationCode: "",
});

const documentForm = reactive({
  type: "link",
  name: "",
  url: "",
  prefix: "",
  file: null,
  content: "",
});

// 获取队列状态
const fetchQueueStatus = async () => {
  try {
    const response = await knowledgeAPI.getQueueStatus();
    if (response.status === 200) {
      queueStatus.value = response.data;
      
      // 根据队列中的文档更新按钮状态
      if (queueStatus.value.processing_documents) {
        processingDocuments.value.clear();
        graphingDocuments.value.clear();
        
        queueStatus.value.processing_documents.forEach(doc => {
          if (doc.mode === 'vectorize') {
            processingDocuments.value.add(doc.document_id);
          } else if (doc.mode === 'graph') {
            graphingDocuments.value.add(doc.document_id);
          } else {
            // mode === 'all'，同时标记两个
            processingDocuments.value.add(doc.document_id);
            graphingDocuments.value.add(doc.document_id);
          }
        });
      }
      
      // 如果有选中的知识库，更新文档状态
      if (selectedLibrary.value) {
        const detailResponse = await knowledgeAPI.getLibraryDetail(selectedLibrary.value.id);
        if (detailResponse.status === 200) {
          selectedLibrary.value.documents = detailResponse.data.documents;
        }
      }
    }
  } catch (err) {
    console.error('获取队列状态失败:', err);
  }
};

// 启动队列状态轮询
const startQueuePolling = () => {
  if (queueStatusInterval) return;
  
  fetchQueueStatus(); // 立即获取一次
  queueStatusInterval = setInterval(fetchQueueStatus, 3000); // 每 3 秒轮询
};

// 停止队列状态轮询
const stopQueuePolling = () => {
  if (queueStatusInterval) {
    clearInterval(queueStatusInterval);
    queueStatusInterval = null;
  }
};

// 加载知识库列表
const loadLibraries = async () => {
  try {
    loading.value = true;
    error.value = "";
    const response = await knowledgeAPI.getLibraries();

    if (response.status === 200) {
      documentLibraries.value = response.data || [];
    } else {
      throw new Error(response.msg || "获取知识库列表失败");
    }
  } catch (err) {
    console.error("加载知识库列表失败:", err);
    error.value = err.message || "加载知识库列表失败";
    ElMessage.error(error.value);
  } finally {
    loading.value = false;
  }
};

// 选择知识库并加载详情
const selectLibrary = async (library) => {
  try {
    selectedLibrary.value = library;
    documentLoading.value = true;

    const response = await knowledgeAPI.getLibraryDetail(library.id);
    if (response.status === 200) {
      selectedLibrary.value = response.data;
    } else {
      throw new Error(response.msg || "获取知识库详情失败");
    }
  } catch (err) {
    console.error("加载知识库详情失败:", err);
    ElMessage.error(err.message || "加载知识库详情失败");
  } finally {
    documentLoading.value = false;
  }
};

// 创建知识库
const createLibrary = async () => {
  try {
    if (!libraryForm.verificationCode) {
      ElMessage.error("请输入校验码");
      return;
    }
    if (libraryForm.verificationCode !== FRONTEND_VERIFY_CODE) {
      ElMessage.error("校验码错误");
      return;
    }
    loading.value = true;
    const response = await knowledgeAPI.createLibrary({
      title: libraryForm.title,
      description: libraryForm.description,
    });

    if (response.status === 200) {
      ElMessage.success("文档库创建成功");
      closeDialog();
      await loadLibraries();
    } else {
      throw new Error(response.msg || "创建文档库失败");
    }
  } catch (err) {
    if (err === "cancel") {
      return;
    }
    console.error("创建知识库失败:", err);
    ElMessage.error(err.message || "创建文档库失败");
  } finally {
    loading.value = false;
  }
};

// 编辑知识库
const editLibrary = (library) => {
  libraryForm.title = library.title;
  libraryForm.description = library.description;
  selectedLibrary.value = library;
  showEditDialog.value = true;
};

// 更新知识库
const updateLibrary = async () => {
  try {
    loading.value = true;
    const response = await knowledgeAPI.updateLibrary(
      selectedLibrary.value.id,
      {
        title: libraryForm.title,
        description: libraryForm.description,
      }
    );

    if (response.status === 200) {
      ElMessage.success("文档库更新成功");
      closeDialog();
      await loadLibraries();
      // 如果当前选中的是被更新的库，重新加载详情
      if (selectedLibrary.value) {
        await selectLibrary(selectedLibrary.value);
      }
    } else {
      throw new Error(response.msg || "更新文档库失败");
    }
  } catch (err) {
    console.error("更新知识库失败:", err);
    ElMessage.error(err.message || "更新文档库失败");
  } finally {
    loading.value = false;
  }
};

// 删除知识库
const deleteLibrary = async (libraryId) => {
  try {
    await ElMessageBox.confirm(
      "确定要删除这个文档库吗？删除后无法恢复。",
      "删除确认",
      {
        confirmButtonText: "确定",
        cancelButtonText: "取消",
        type: "warning",
      }
    );

    loading.value = true;
    const response = await knowledgeAPI.deleteLibrary(libraryId);

    if (response.status === 200) {
      ElMessage.success("文档库删除成功");
      if (selectedLibrary.value && selectedLibrary.value.id === libraryId) {
        selectedLibrary.value = null;
      }
      await loadLibraries();
    } else {
      throw new Error(response.msg || "删除文档库失败");
    }
  } catch (err) {
    if (err !== "cancel") {
      console.error("删除知识库失败:", err);
      ElMessage.error(err.message || "删除文档库失败");
    }
  } finally {
    loading.value = false;
  }
};

// 添加文档
const addDocument = async () => {
  try {
    if (!selectedLibrary.value) return;

    loading.value = true;

    // 如果是网站链接类型，使用爬取接口
    if (documentForm.type === "link" && documentForm.url) {
      try {
        // 计算prefix：如果用户没有填写，则使用URL最后一个斜杠及之前的内容
        let prefix = documentForm.prefix;
        if (!prefix) {
          const url = new URL(documentForm.url);
          const pathname = url.pathname;
          const lastSlashIndex = pathname.lastIndexOf("/");
          if (lastSlashIndex > 0) {
            prefix = url.origin + pathname.substring(0, lastSlashIndex + 1);
          } else {
            prefix = url.origin + "/";
          }
        }

        const crawlData = {
          url: documentForm.url,
          prefix: prefix,
          title: documentForm.name,
          collection_id: selectedLibrary.value.collection_id,
          user_id: "default_user", // 可以根据实际情况设置用户ID
          if_llm: false, // 可以根据需要设置是否使用LLM
        };

        const crawlResponse = await knowledgeAPI.crawlSite(crawlData);

        if (crawlResponse.status === 200) {
          ElMessage.success("网站爬取任务已启动，正在后台处理...");
          resetDocumentForm();
          showAddDocumentDialog.value = false;
          // 重新加载当前知识库详情
          await selectLibrary(selectedLibrary.value);
          return;
        } else {
          throw new Error(crawlResponse.msg || "网站爬取失败");
        }
      } catch (crawlError) {
        console.error("网站爬取失败:", crawlError);
        ElMessage.error("网站爬取失败: " + crawlError.message);
        return;
      }
    }

    // 原有的文档添加逻辑（用于文件上传）
    let documentData = {
      library_id: selectedLibrary.value.id,
      name: documentForm.name,
      type: documentForm.type,
      content: documentForm.content || "",
      url: documentForm.url || "",
    };

    // 如果是文件上传类型，需要先上传文件到OSS
    if (documentForm.type === "file" && documentForm.file) {
      try {
        // 1. 获取OSS上传签名URL
        const uploadResponse = await knowledgeAPI.getUploadUrl(
          documentForm.file.name
        );

        if (uploadResponse.status !== 200 || !uploadResponse.data) {
          throw new Error("获取上传URL失败");
        }

        // 2. 上传文件到OSS（支持返回对象形式，避免签名不匹配）
        await knowledgeAPI.uploadFileToOSS(uploadResponse.data, documentForm.file);

        // 3. 设置文档的URL为OSS文件路径
        const presign = uploadResponse.data;
        const presignUrl = typeof presign === "string" ? presign : presign.url;
        documentData.url = presignUrl.split("?")[0];
        documentData.content = ""; // 文件上传后不需要content字段

        ElMessage.success("文件上传成功");
      } catch (uploadError) {
        console.error("文件上传失败:", uploadError);
        ElMessage.error("文件上传失败: " + uploadError.message);
        return;
      }
    }

    const response = await knowledgeAPI.addDocument(documentData);

    if (response.status === 200) {
      ElMessage.success("文档添加成功");
      resetDocumentForm();
      showAddDocumentDialog.value = false;
      // 重新加载当前知识库详情
      await selectLibrary(selectedLibrary.value);
    } else {
      throw new Error(response.msg || "添加文档失败");
    }
  } catch (err) {
    console.error("添加文档失败:", err);
    ElMessage.error(err.message || "添加文档失败");
  } finally {
    loading.value = false;
  }
};

// 查看知识图谱
const viewKnowledgeGraph = () => {
  if (selectedLibrary.value) {
    router.push({
      name: "knowledge-graph",
      params: { collection_id: selectedLibrary.value.collection_id },
    });
  }
};

// 删除文档
const removeDocument = async (documentId) => {
  try {
    await ElMessageBox.confirm("确定要移除这个文档吗？", "移除确认", {
      confirmButtonText: "确定",
      cancelButtonText: "取消",
      type: "warning",
    });

    loading.value = true;
    const response = await knowledgeAPI.deleteDocument(documentId);

    if (response.status === 200) {
      ElMessage.success("文档移除成功");
      // 重新加载当前知识库详情
      await selectLibrary(selectedLibrary.value);
    } else {
      throw new Error(response.msg || "移除文档失败");
    }
  } catch (err) {
    if (err !== "cancel") {
      console.error("删除文档失败:", err);
      ElMessage.error(err.message || "移除文档失败");
    }
  } finally {
    loading.value = false;
  }
};

// 处理文件选择
const handleFileSelect = (event) => {
  const file = event.target.files[0];
  if (file) {
    documentForm.file = file;
    if (!documentForm.name) {
      documentForm.name = file.name;
    }

    // 对于文件上传，不需要读取文件内容到content字段
    // 文件将直接上传到OSS
    documentForm.content = "";
  }
};

// 重置文档表单
const resetDocumentForm = () => {
  documentForm.name = "";
  documentForm.url = "";
  documentForm.prefix = "";
  documentForm.file = null;
  documentForm.content = "";
  documentForm.type = "link";
};

// 关闭对话框
const closeDialog = () => {
  showCreateDialog.value = false;
  showEditDialog.value = false;
  libraryForm.title = "";
  libraryForm.description = "";
  libraryForm.verificationCode = "";
};

// 获取文档类型样式
const getDocumentTypeClass = (type) => {
  switch (type) {
    case "pdf":
      return "bg-red-100 text-red-600";
    case "link":
      return "bg-blue-100 text-blue-600";
    case "file":
      return "bg-green-100 text-green-600";
    default:
      return "bg-gray-100 text-gray-600";
  }
};

// 格式化时间
const formatTime = (date) => {
  if (!date) return "";
  const now = new Date();
  const targetDate = new Date(date);
  const diffInHours = (now - targetDate) / (1000 * 60 * 60);

  if (diffInHours < 1) {
    return "刚刚";
  } else if (diffInHours < 24) {
    return `${Math.floor(diffInHours)}小时前`;
  } else {
    return targetDate.toLocaleDateString("zh-CN");
  }
};

// 文档预览相关
const showPreviewDialog = ref(false);
const previewLoading = ref(false);
const previewData = reactive({
  id: null,
  name: "",
  type: "",
  url: "",
  content: "",
  content_type: "",
});

// 文档分块查看相关
const showChunksDialog = ref(false);
const chunksLoading = ref(false);
const chunksData = reactive({
  document_id: null,
  document_name: "",
  total_chunks: 0,
  chunks: [],
  message: ""
});

// 正在处理的文档集合（用于显示加载状态） - 已移动到上方，此处删除

// 开始向量解析（简化版，依赖队列轮询）
const startVectorize = async (documentId) => {
  try {
    processingDocuments.value.add(documentId);
    
    const response = await knowledgeAPI.startDocumentVectorize(documentId);
    
    if (response.status === 200) {
      ElMessage.success(response.data?.message || "文档已加入向量化队列");
      startQueuePolling(); // 启动队列轮询
    } else {
      throw new Error(response.msg || "开始向量化失败");
    }
  } catch (err) {
    console.error("开始向量化失败:", err);
    ElMessage.error(err.message || "开始向量化失败");
    processingDocuments.value.delete(documentId);
  }
};

// 开始图谱解析（简化版，依赖队列轮询）
const startGraph = async (documentId) => {
  try {
    graphingDocuments.value.add(documentId);
    
    const response = await knowledgeAPI.startDocumentGraph(documentId);
    
    if (response.status === 200) {
      ElMessage.success(response.data?.message || "文档已加入图谱化队列");
      startQueuePolling(); // 启动队列轮询
    } else {
      throw new Error(response.msg || "开始图谱化失败");
    }
  } catch (err) {
    console.error("开始图谱化失败:", err);
    ElMessage.error(err.message || "开始图谱化失败");
    graphingDocuments.value.delete(documentId);
  }
};

// 开始解析文档（向量化 + 图谱化）
const startProcessing = async (documentId) => {
  try {
    processingDocuments.value.add(documentId);
    
    const response = await knowledgeAPI.startDocumentProcessing(documentId);
    
    if (response.status === 200) {
      ElMessage.success(response.data?.message || "文档已加入解析队列");
      
      // 每隔 5 秒检查一次解析状态（降低刷新频率）
      const checkInterval = setInterval(async () => {
        try {
          // 只获取当前知识库详情，更新状态
          const detailResponse = await knowledgeAPI.getLibraryDetail(selectedLibrary.value.id);
          if (detailResponse.status === 200) {
            // 只更新文档列表，不触发整个页面刷新
            selectedLibrary.value.documents = detailResponse.data.documents;
            
            // 检查文档是否已解析完成
            const doc = selectedLibrary.value.documents?.find(d => d.id === documentId);
            if (doc && doc.is_processed) {
              clearInterval(checkInterval);
              processingDocuments.value.delete(documentId);
              ElMessage.success(`文档「${doc.name}」解析完成`);
            }
          }
        } catch (err) {
          console.error("检查文档状态失败:", err);
        }
      }, 50000);
      
      // 60秒后停止轮询（防止无限轮询）
      setTimeout(() => {
        clearInterval(checkInterval);
        processingDocuments.value.delete(documentId);
      }, 60000);
    } else {
      throw new Error(response.msg || "开始解析失败");
    }
  } catch (err) {
    console.error("开始解析失败:", err);
    ElMessage.error(err.message || "开始解析失败");
    processingDocuments.value.delete(documentId);
  }
};

// 一键向量化：将所有未向量化的文档加入队列
const batchVectorize = async () => {
  if (!selectedLibrary.value || !selectedLibrary.value.documents) return;
  
  const unvectorizedDocs = selectedLibrary.value.documents.filter(doc => 
    doc.type === 'file' && !doc.is_vectorized
  );
  
  if (unvectorizedDocs.length === 0) {
    ElMessage.warning('没有需要向量化的文档');
    return;
  }
  
  try {
    await ElMessageBox.confirm(
      `确定要将 ${unvectorizedDocs.length} 个文档加入向量化队列吗？`,
      '批量向量化',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'info'
      }
    );
    
    loading.value = true;
    let successCount = 0;
    
    for (const doc of unvectorizedDocs) {
      try {
        const response = await knowledgeAPI.startDocumentVectorize(doc.id);
        if (response.status === 200) {
          processingDocuments.value.add(doc.id);
          successCount++;
        }
      } catch (err) {
        console.error(`文档 ${doc.name} 向量化失败:`, err);
      }
    }
    
    ElMessage.success(`成功加入 ${successCount} 个文档到向量化队列`);
    startQueuePolling();
  } catch (err) {
    if (err !== 'cancel') {
      console.error('批量向量化失败:', err);
    }
  } finally {
    loading.value = false;
  }
};

// 一键图谱化：将所有未图谱化的文档加入队列
const batchGraph = async () => {
  if (!selectedLibrary.value || !selectedLibrary.value.documents) return;
  
  const ungraphedDocs = selectedLibrary.value.documents.filter(doc => 
    doc.type === 'file' && !doc.is_graphed
  );
  
  if (ungraphedDocs.length === 0) {
    ElMessage.warning('没有需要图谱化的文档');
    return;
  }
  
  try {
    await ElMessageBox.confirm(
      `确定要将 ${ungraphedDocs.length} 个文档加入图谱化队列吗？`,
      '批量图谱化',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'info'
      }
    );
    
    loading.value = true;
    let successCount = 0;
    
    for (const doc of ungraphedDocs) {
      try {
        const response = await knowledgeAPI.startDocumentGraph(doc.id);
        if (response.status === 200) {
          graphingDocuments.value.add(doc.id);
          successCount++;
        }
      } catch (err) {
        console.error(`文档 ${doc.name} 图谱化失败:`, err);
      }
    }
    
    ElMessage.success(`成功加入 ${successCount} 个文档到图谱化队列`);
    startQueuePolling();
  } catch (err) {
    if (err !== 'cancel') {
      console.error('批量图谱化失败:', err);
    }
  } finally {
    loading.value = false;
  }
};

// 预览文档
const previewDocument = async (document) => {
  try {
    showPreviewDialog.value = true;
    previewLoading.value = true;

    // 重置预览数据
    previewData.id = document.id;
    previewData.name = document.name;
    previewData.type = document.type;
    previewData.url = document.url;
    previewData.content = "";
    previewData.content_type = "";

    const response = await knowledgeAPI.getDocumentContent(document.id);

    if (response.status === 200 && response.data) {
      previewData.content = response.data.content || "";
      previewData.content_type = response.data.content_type || "";
    } else {
      ElMessage.warning(response.msg || "无法获取文档内容");
    }
  } catch (error) {
    console.error("预览文档失败:", error);
    ElMessage.error("预览文档失败: " + (error.message || "未知错误"));
  } finally {
    previewLoading.value = false;
  }
};

// 查看文档分块
const viewDocumentChunks = async (document) => {
  try {
    showChunksDialog.value = true;
    chunksLoading.value = true;

    // 重置分块数据
    chunksData.document_id = document.id;
    chunksData.document_name = document.name;
    chunksData.total_chunks = 0;
    chunksData.chunks = [];
    chunksData.message = "";

    const response = await knowledgeAPI.getDocumentChunks(document.id);

    if (response.status === 200 && response.data) {
      chunksData.total_chunks = response.data.total_chunks || 0;
      chunksData.chunks = response.data.chunks || [];
      chunksData.message = response.data.message || "";
    } else {
      ElMessage.warning(response.msg || "无法获取分块信息");
    }
  } catch (error) {
    console.error("获取分块信息失败:", error);
    ElMessage.error("获取分块信息失败: " + (error.message || "未知错误"));
  } finally {
    chunksLoading.value = false;
  }
};

// 渲染文本（用于预览弹窗）
const renderPreviewMarkdown = (content) => {
  if (!content) return "";
  try {
    return marked.parse(content);
  } catch (error) {
    console.error("文本渲染失败:", error);
    return `<pre>${content}</pre>`;
  }
};

// 组件挂载时加载数据
onMounted(() => {
  loadLibraries();
  startQueuePolling(); // 启动队列轮询
});

onUnmounted(() => {
  stopQueuePolling(); // 组件卸载时停止轮询
});
</script>

<style scoped>
.line-clamp-2 {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* 抽屉滑入动画 */
@keyframes slideInRight {
  from {
    transform: translateX(100%);
  }
  to {
    transform: translateX(0);
  }
}

.animate-slide-in-right {
  animation: slideInRight 0.3s ease-out;
}
</style>
