<script setup lang="ts">
import { ref } from 'vue'
import { useSessionStore } from '../../stores/session'
import { Plus, Trash2, MessageSquare } from 'lucide-vue-next'

const store = useSessionStore()
const modes = [
  { value: 'privacy_enhanced', label: '脱敏上传' },
  { value: 'local_fallback', label: '本地处理' },
]
// ponytail: 仅两种隐私档位，会话级模式，默认脱敏
const newMode = ref('privacy_enhanced')

const editingId = ref('')
const editingTitle = ref('')

function startEdit(s: { session_id: string; title: string }) {
  editingId.value = s.session_id
  editingTitle.value = s.title
}

function commitEdit() {
  const t = editingTitle.value.trim()
  if (t && editingId.value) {
    store.rename(editingId.value, t)
  }
  editingId.value = ''
}

function cancelEdit() {
  editingId.value = ''
}
</script>

<template>
  <aside class="w-64 border-r border-slate-200/60 bg-white/40 flex flex-col">
    <div class="p-3 space-y-2">
      <div class="flex rounded-lg border border-slate-200 bg-white p-0.5 text-xs">
        <button
          v-for="m in modes"
          :key="m.value"
          @click="newMode = m.value"
          class="flex-1 rounded-md py-1.5 font-medium transition"
          :class="newMode === m.value ? 'bg-primary text-white' : 'text-slate-500 hover:text-primary'"
        >
          {{ m.label }}
        </button>
      </div>
      <button @click="store.create(newMode)" class="w-full rounded-lg border border-primary/30 text-primary py-2 text-sm font-medium hover:bg-primary/5 flex items-center justify-center gap-1.5 transition">
        <Plus class="w-4 h-4" /> 新建会话
      </button>
    </div>
    <div class="flex-1 overflow-y-auto px-2 space-y-1">
      <button
        v-for="s in store.sessions"
        :key="s.session_id"
        @click="store.select(s.session_id)"
        class="group w-full text-left rounded-lg px-3 py-2.5 flex items-center gap-2 transition"
        :class="s.session_id === store.currentId ? 'bg-primary/10 text-primary' : 'hover:bg-slate-100 text-slate-600'"
      >
        <MessageSquare class="w-4 h-4 flex-shrink-0" />
        <input
          v-if="editingId === s.session_id"
          v-model="editingTitle"
          @keydown.enter="commitEdit"
          @keydown.esc="cancelEdit"
          @blur="commitEdit"
          @click.stop
          class="flex-1 min-w-0 text-sm bg-white border border-primary rounded px-1 py-0.5 outline-none"
        />
        <template v-else>
          <span class="flex-1 truncate text-sm" @dblclick.stop="startEdit(s)">{{ s.title }}</span>
          <span
            class="flex-shrink-0 text-[10px] px-1.5 py-0.5 rounded font-medium leading-none"
            :class="s.mode === 'privacy_enhanced' ? 'bg-indigo-50 text-indigo-500' : 'bg-amber-50 text-amber-600'"
          >{{ s.mode === 'privacy_enhanced' ? '脱' : '本' }}</span>
        </template>
        <span @click.stop="store.remove(s.session_id)" class="opacity-0 group-hover:opacity-100 hover:text-red-500 transition">
          <Trash2 class="w-3.5 h-3.5" />
        </span>
      </button>
      <div v-if="!store.sessions.length" class="text-center text-xs text-slate-400 py-8">暂无会话</div>
    </div>
  </aside>
</template>
