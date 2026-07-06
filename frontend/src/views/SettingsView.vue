<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getSettings, updateSettings, type Settings } from '../api/settings'
import { Loader2, Check } from 'lucide-vue-next'

const settings = ref<Settings | null>(null)
const saving = ref(false)
const saved = ref(false)
const error = ref('')

// 表单字段
const modelName = ref('')
const modelBaseUrl = ref('')
const apiKey = ref('')
const fallbackEnabled = ref(true)
const fallbackOllamaBaseUrl = ref('')
const fallbackOllamaModel = ref('')

// 预设模型
const presets = [
  { name: 'deepseek-chat', url: 'https://api.deepseek.com/v1', label: 'DeepSeek Chat' },
  { name: 'deepseek-v4-flash', url: 'https://api.deepseek.com/v1', label: 'DeepSeek Flash' },
  { name: 'gpt-4o', url: 'https://api.openai.com/v1', label: 'GPT-4o' },
  { name: 'gpt-4o-mini', url: 'https://api.openai.com/v1', label: 'GPT-4o Mini' },
  { name: 'qwen-plus', url: 'https://dashscope.aliyuncs.com/compatible-mode/v1', label: '通义千问 Plus' },
  { name: 'glm-4-flash', url: 'https://open.bigmodel.cn/api/paas/v4', label: '智谱 GLM-4 Flash' },
]

onMounted(async () => {
  try {
    const s = await getSettings()
    settings.value = s
    modelName.value = s.model_name
    modelBaseUrl.value = s.model_base_url
    fallbackEnabled.value = s.fallback_enabled
    fallbackOllamaBaseUrl.value = s.fallback_ollama_base_url
    fallbackOllamaModel.value = s.fallback_ollama_model
  } catch (e) {
    error.value = '加载设置失败'
  }
})

function applyPreset(preset: typeof presets[0]) {
  modelName.value = preset.name
  modelBaseUrl.value = preset.url
}

async function save() {
  saving.value = true
  error.value = ''
  try {
    const payload: Record<string, unknown> = {}
    if (modelName.value !== settings.value?.model_name) payload.model_name = modelName.value
    if (modelBaseUrl.value !== settings.value?.model_base_url) payload.model_base_url = modelBaseUrl.value
    if (apiKey.value) payload.api_key = apiKey.value
    if (fallbackEnabled.value !== settings.value?.fallback_enabled) payload.fallback_enabled = fallbackEnabled.value
    if (fallbackOllamaBaseUrl.value !== settings.value?.fallback_ollama_base_url) payload.fallback_ollama_base_url = fallbackOllamaBaseUrl.value
    if (fallbackOllamaModel.value !== settings.value?.fallback_ollama_model) payload.fallback_ollama_model = fallbackOllamaModel.value

    const s = await updateSettings(payload)
    settings.value = s
    apiKey.value = ''
    saved.value = true
    setTimeout(() => (saved.value = false), 2000)
  } catch (e) {
    error.value = '保存失败'
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <div class="h-full overflow-y-auto px-6 py-6">
    <div class="max-w-2xl mx-auto space-y-8">
      <h1 class="text-xl font-semibold text-slate-800">设置</h1>

      <!-- 模型选择 -->
      <section class="space-y-4">
        <h2 class="text-sm font-medium text-slate-500 uppercase tracking-wide">主模型</h2>
        <div class="flex flex-wrap gap-2">
          <button
            v-for="p in presets"
            :key="p.name"
            @click="applyPreset(p)"
            class="px-3 py-1.5 text-xs rounded-lg border transition"
            :class="modelName === p.name && modelBaseUrl === p.url
              ? 'border-primary bg-primary/10 text-primary'
              : 'border-slate-200 text-slate-600 hover:border-primary/50'"
          >{{ p.label }}</button>
        </div>
        <div class="grid gap-3">
          <label class="block">
            <span class="text-xs text-slate-500">模型名称</span>
            <input v-model="modelName" class="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-primary" />
          </label>
          <label class="block">
            <span class="text-xs text-slate-500">Base URL</span>
            <input v-model="modelBaseUrl" class="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-primary" />
          </label>
          <label class="block">
            <span class="text-xs text-slate-500">API Key（留空则不更新）</span>
            <input v-model="apiKey" type="password" placeholder="sk-..." class="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-primary" />
          </label>
          <p v-if="settings?.api_key_masked" class="text-xs text-slate-400">当前 Key：{{ settings.api_key_masked }}</p>
        </div>
      </section>

      <!-- 本地模型 -->
      <section class="space-y-4">
        <h2 class="text-sm font-medium text-slate-500 uppercase tracking-wide">本地模型（Ollama）</h2>
        <label class="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" v-model="fallbackEnabled" class="rounded border-slate-300" />
          <span class="text-sm text-slate-700">启用本地模型兜底</span>
        </label>
        <div class="grid gap-3" :class="{ 'opacity-50': !fallbackEnabled }">
          <label class="block">
            <span class="text-xs text-slate-500">Ollama Base URL</span>
            <input v-model="fallbackOllamaBaseUrl" :disabled="!fallbackEnabled" class="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-primary disabled:bg-slate-50" />
          </label>
          <label class="block">
            <span class="text-xs text-slate-500">Ollama 模型名称</span>
            <input v-model="fallbackOllamaModel" :disabled="!fallbackEnabled" class="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm outline-none focus:border-primary disabled:bg-slate-50" />
          </label>
        </div>
      </section>

      <!-- 保存 -->
      <div class="flex items-center gap-3">
        <button
          @click="save"
          :disabled="saving"
          class="px-5 py-2 rounded-lg bg-primary text-white text-sm font-medium hover:bg-primary-light disabled:opacity-50 transition flex items-center gap-2"
        >
          <Loader2 v-if="saving" class="w-4 h-4 animate-spin" />
          <Check v-else-if="saved" class="w-4 h-4" />
          {{ saving ? '保存中...' : saved ? '已保存' : '保存设置' }}
        </button>
        <span v-if="error" class="text-sm text-red-500">{{ error }}</span>
      </div>
    </div>
  </div>
</template>
