<script setup lang="ts">
import { ref, nextTick, watch, computed } from 'vue'
import { useSessionStore } from '../stores/session'
import SessionSidebar from '../components/chat/SessionSidebar.vue'
import MessageBubble from '../components/chat/MessageBubble.vue'
import { Send, Paperclip, Loader2, MessageSquare, Coins } from 'lucide-vue-next'

const store = useSessionStore()
const input = ref('')
const fileInput = ref<HTMLInputElement | null>(null)
const scrollRef = ref<HTMLDivElement | null>(null)
const toast = ref('')

const modes = [
  { value: 'privacy_enhanced', label: '脱敏上传' },
  { value: 'local_fallback', label: '本地处理' },
]
const currentSession = computed(() =>
  store.sessions.find((s) => s.session_id === store.currentId) || null,
)

store.loadSessions().catch((e) => console.error('加载会话失败', e))

async function scrollToBottom() {
  await nextTick()
  if (scrollRef.value) scrollRef.value.scrollTop = scrollRef.value.scrollHeight
}

watch(() => store.messages.length, scrollToBottom)
watch(() => store.messages[store.messages.length - 1]?.content, scrollToBottom)

async function send() {
  const msg = input.value.trim()
  if (!msg || store.streaming) return
  input.value = ''
  await store.send(msg)
}

function onFileChange(e: Event) {
  const el = e.target as HTMLInputElement
  if (!el.files || !el.files.length) return
  store.uploadFiles(Array.from(el.files)).then((n) => {
    toast.value = `已索引 ${n} 个分块`
    setTimeout(() => (toast.value = ''), 2000)
  }).catch((e) => console.error('上传失败', e))
  el.value = ''
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    send()
  }
}
</script>

<template>
  <div class="h-full flex">
    <SessionSidebar />
    <section class="flex-1 flex flex-col">
      <!-- 顶部：当前会话隐私模式切换 -->
      <header v-if="currentSession" class="flex items-center justify-between border-b border-slate-200/60 bg-white/60 px-6 py-3">
        <div class="min-w-0">
          <p class="truncate text-sm font-medium text-slate-700">{{ currentSession.title }}</p>
          <p class="text-xs text-slate-400">隐私模式</p>
        </div>
        <div class="flex rounded-lg border border-slate-200 bg-white p-0.5 text-xs">
          <button
            v-for="m in modes"
            :key="m.value"
            @click="store.setMode(m.value)"
            class="rounded-md px-3 py-1.5 font-medium transition"
            :class="currentSession.mode === m.value ? 'bg-primary text-white' : 'text-slate-500 hover:text-primary'"
          >
            {{ m.label }}
          </button>
        </div>
      </header>
      <!-- 消息流 -->
      <div ref="scrollRef" class="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        <div v-if="!store.messages.length" class="h-full flex items-center justify-center text-slate-400">
          <div class="text-center">
            <MessageSquare class="w-12 h-12 mx-auto mb-3 opacity-40" />
            <p class="text-sm">开始一段新的对话</p>
          </div>
        </div>
        <MessageBubble
          v-for="(m, i) in store.messages"
          :key="i"
          :message="m"
          :streaming="store.streaming && i === store.messages.length - 1"
        />
      </div>

      <!-- toast -->
      <div v-if="toast" class="px-6 text-xs text-emerald-600">{{ toast }}</div>

      <!-- 输入区 -->
      <div class="border-t border-slate-200/60 bg-white/60 p-4">
        <div class="flex items-end gap-2 rounded-xl border border-slate-200 bg-white p-2 focus-within:border-primary transition">
          <button class="p-2 text-slate-400 hover:text-primary rounded-lg hover:bg-slate-50" @click="fileInput?.click()">
            <Paperclip class="w-5 h-5" />
          </button>
          <input ref="fileInput" type="file" multiple class="hidden" @change="onFileChange" />
          <textarea
            v-model="input"
            @keydown="onKeydown"
            rows="1"
            class="flex-1 resize-none outline-none text-sm py-2 max-h-32"
            placeholder="输入消息，Enter 发送，Shift+Enter 换行"
          ></textarea>
          <button
            @click="send"
            :disabled="store.streaming || !input.trim()"
            class="p-2 rounded-lg bg-primary text-white disabled:opacity-40 hover:bg-primary-light transition"
          >
            <Loader2 v-if="store.streaming" class="w-5 h-5 animate-spin" />
            <Send v-else class="w-5 h-5" />
          </button>
        </div>
      </div>
      <!-- 费用统计 -->
      <div v-if="store.cost" class="border-t border-slate-100 bg-slate-50/80 px-6 py-2 flex items-center gap-4 text-xs text-slate-400">
        <span class="flex items-center gap-1">
          <Coins class="w-3.5 h-3.5" />
          本次会话：<span class="text-slate-600 font-medium">¥{{ store.cost.cost_yuan.toFixed(4) }}</span>
        </span>
        <span>Token：{{ store.cost.tokens_in }} / {{ store.cost.tokens_out }}</span>
        <span v-if="store.cost.balance_yuan !== null">
          余额：<span class="text-emerald-600 font-medium">¥{{ store.cost.balance_yuan.toFixed(2) }}</span>
        </span>
        <span v-else>
          预算：<span class="text-slate-600">¥{{ store.cost.budget_yuan.toFixed(2) }}</span>
        </span>
      </div>
    </section>
  </div>
</template>
