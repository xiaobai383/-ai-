<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { RouterLink, RouterView } from 'vue-router'
import { MessageSquare, Settings } from 'lucide-vue-next'
import { http } from '../../api/request'

const healthy = ref(false)
onMounted(() => {
  http
    .get('/health')
    .then(() => (healthy.value = true))
    .catch((e) => {
      console.error('健康检查失败', e)
      healthy.value = false
    })
})
</script>

<template>
  <div class="flex flex-col h-screen">
    <!-- 顶部导航 -->
    <header class="fixed top-0 inset-x-0 h-14 z-50 glass-card flex items-center px-6">
      <div class="flex items-center gap-2">
        <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-primary-light flex items-center justify-center shadow-sm">
          <MessageSquare class="w-5 h-5 text-white" />
        </div>
        <span class="text-lg font-semibold text-slate-800">Hush</span>
      </div>
      <div class="ml-auto flex items-center gap-2 text-sm">
        <span class="flex items-center gap-1.5" :class="healthy ? 'text-emerald-500' : 'text-slate-400'">
          <span class="w-2 h-2 rounded-full" :class="healthy ? 'bg-emerald-500' : 'bg-slate-300'"></span>
          {{ healthy ? '服务正常' : '未连接' }}
        </span>
      </div>
    </header>

    <div class="flex flex-1 pt-14 h-screen">
      <!-- 侧边导航 -->
      <aside class="w-20 flex flex-col items-center gap-2 py-6 border-r border-slate-200/60 bg-white/50">
        <RouterLink to="/" v-slot="{ isActive }">
          <div class="flex flex-col items-center gap-1 px-3 py-2 rounded-lg transition cursor-pointer" :class="isActive ? 'bg-primary text-white shadow-sm' : 'text-slate-500 hover:bg-slate-100'">
            <MessageSquare class="w-5 h-5" />
            <span class="text-xs">对话</span>
          </div>
        </RouterLink>
        <RouterLink to="/settings" v-slot="{ isActive }">
          <div class="flex flex-col items-center gap-1 px-3 py-2 rounded-lg transition cursor-pointer" :class="isActive ? 'bg-primary text-white shadow-sm' : 'text-slate-500 hover:bg-slate-100'">
            <Settings class="w-5 h-5" />
            <span class="text-xs">设置</span>
          </div>
        </RouterLink>
      </aside>

      <!-- 主内容 -->
      <main class="flex-1 overflow-hidden">
        <RouterView />
      </main>
    </div>
  </div>
</template>
