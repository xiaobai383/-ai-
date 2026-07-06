<script setup lang="ts">
import { CheckCircle2, XCircle } from 'lucide-vue-next'
import type { StepEvent } from '../../api/types'

const props = defineProps<{ steps: StepEvent[] }>()

const stepLabels: Record<string, string> = {
  preprocess: '预处理',
  upload_policy: '上传策略',
  limit_check: '限额检查',
  llm_call: 'LLM 调用',
  local_analysis: '本地分析',
  postprocess: '后处理',
  save_result: '保存结果',
  redact_input: '脱敏输入',
  retrieve: '检索',
}
</script>

<template>
  <div class="bg-white rounded-lg border border-slate-200 p-4">
    <h3 class="text-sm font-medium text-slate-700 mb-3">执行步骤</h3>
    <div class="relative">
      <div class="absolute left-2 top-2 bottom-2 w-px bg-slate-200"></div>
      <div class="space-y-3">
        <div v-for="s in props.steps" :key="s.step_id" class="flex items-start gap-3 relative">
          <div class="w-4 h-4 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5 z-10" :class="s.status === 'success' ? 'bg-emerald-500' : 'bg-red-500'">
            <CheckCircle2 v-if="s.status === 'success'" class="w-3 h-3 text-white" />
            <XCircle v-else class="w-3 h-3 text-white" />
          </div>
          <div class="flex-1 min-w-0 pb-1">
            <div class="flex items-center gap-2 flex-wrap">
              <span class="text-sm font-medium text-slate-700">{{ stepLabels[s.name] || s.name }}</span>
              <span class="text-xs text-slate-400">{{ s.duration_ms }}ms</span>
              <span v-if="s.tokens_in || s.tokens_out" class="text-xs text-slate-400">· {{ s.tokens_in }}/{{ s.tokens_out }} tok</span>
              <span v-if="s.cost_yuan" class="text-xs text-slate-400">· ¥{{ s.cost_yuan.toFixed(4) }}</span>
            </div>
            <p v-if="s.output_preview" class="text-xs text-slate-500 mt-0.5 truncate">{{ s.output_preview }}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
