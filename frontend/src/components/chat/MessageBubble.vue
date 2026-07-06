<script setup lang="ts">
import { ref } from 'vue'
import { User, Bot, ChevronDown } from 'lucide-vue-next'
import type { Message } from '../../api/types'
import StepTimeline from '../task/StepTimeline.vue'

const props = defineProps<{ message: Message; streaming?: boolean }>()
const showSteps = ref(false)
</script>

<template>
  <div class="flex gap-3" :class="props.message.role === 'user' ? 'flex-row-reverse' : ''">
    <div
      class="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 shadow-sm"
      :class="props.message.role === 'user' ? 'bg-primary text-white' : 'bg-slate-100 text-slate-600'"
    >
      <User v-if="props.message.role === 'user'" class="w-4 h-4" />
      <Bot v-else class="w-4 h-4" />
    </div>
    <div
      class="max-w-[75%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed"
      :class="props.message.role === 'user' ? 'bg-primary text-white rounded-tr-sm' : 'bg-white border border-slate-200 text-slate-700 rounded-tl-sm'"
    >
      <button
        v-if="props.message.role === 'assistant' && props.message.steps?.length"
        @click="showSteps = !showSteps"
        class="flex items-center gap-1 mb-2 text-xs text-slate-400 hover:text-primary transition"
      >
        <ChevronDown class="w-3 h-3 transition-transform" :class="showSteps ? '' : '-rotate-90'" />
        思考过程（{{ props.message.steps.length }} 步）
      </button>
      <StepTimeline v-if="showSteps && props.message.steps" :steps="props.message.steps" class="mb-2" />
      <div class="whitespace-pre-wrap break-words" :class="{ 'stream-cursor': props.streaming }">{{ props.message.content || '...' }}</div>
    </div>
  </div>
</template>
