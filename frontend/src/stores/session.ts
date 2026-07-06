import { defineStore } from 'pinia'
import { ref } from 'vue'
import * as sessionApi from '../api/session'
import type { Session, Message, ChatSSEEvent } from '../api/types'

export interface SessionCost {
  tokens_in: number
  tokens_out: number
  cost_yuan: number
  balance_yuan: number | null
  budget_yuan: number
}

export const useSessionStore = defineStore('session', () => {
  const sessions = ref<Session[]>([])
  const currentId = ref('')
  const messages = ref<Message[]>([])
  const streaming = ref(false)
  const streamingText = ref('')
  const controller = ref<AbortController | null>(null)
  const cost = ref<SessionCost | null>(null)

  async function loadSessions() {
    sessions.value = await sessionApi.listSessions()
    if (sessions.value.length && !currentId.value) {
      await select(sessions.value[0].session_id)
    }
  }

  async function loadCost() {
    if (!currentId.value) { cost.value = null; return }
    try {
      cost.value = await sessionApi.getSessionCost(currentId.value)
    } catch { cost.value = null }
  }

  async function select(id: string) {
    currentId.value = id
    messages.value = await sessionApi.getMessages(id)
    await loadCost()
  }

  async function create(mode: string = 'privacy_enhanced') {
    const s = await sessionApi.createSession('新建会话', mode)
    sessions.value.unshift(s)
    await select(s.session_id)
  }

  async function remove(id: string) {
    await sessionApi.deleteSession(id)
    sessions.value = sessions.value.filter((s) => s.session_id !== id)
    if (currentId.value === id) {
      currentId.value = ''
      messages.value = []
      if (sessions.value.length) await select(sessions.value[0].session_id)
    }
  }

  async function rename(id: string, title: string) {
    const s = await sessionApi.renameSession(id, title)
    const idx = sessions.value.findIndex((x) => x.session_id === id)
    if (idx >= 0) sessions.value[idx] = s
  }

  async function setMode(mode: string) {
    if (!currentId.value) return
    const s = await sessionApi.updateSessionMode(currentId.value, mode)
    const idx = sessions.value.findIndex((x) => x.session_id === currentId.value)
    if (idx >= 0) sessions.value[idx] = s
  }

  async function uploadFiles(files: File[]): Promise<number> {
    if (!currentId.value) return 0
    return sessionApi.uploadSessionFiles(currentId.value, files)
  }

  async function send(message: string) {
    if (!currentId.value || streaming.value) return
    messages.value.push({ role: 'user', content: message })
    messages.value.push({ role: 'assistant', content: '', steps: [] })
    streaming.value = true
    streamingText.value = ''
    controller.value = new AbortController()
    await sessionApi
      .sendMessageStream(
        currentId.value,
        message,
        (evt: ChatSSEEvent) => {
          const last = messages.value[messages.value.length - 1]
          if (evt.type === 'step') {
            // 思考过程步骤：累积到当前助手消息
            last.steps = [...(last.steps || []), evt]
          } else if (evt.type === 'token') {
            // 后端推送的是已还原占位符的完整文本，直接替换（非追加）
            streamingText.value = evt.text
            last.content = streamingText.value
          } else if (evt.type === 'done') {
            last.content = evt.reply
            streaming.value = false
            loadSessions()
            loadCost()
          } else if (evt.type === 'error') {
            last.content = `❌ ${evt.message}`
            streaming.value = false
          }
        },
        controller.value.signal,
      )
      .catch((e) => {
        messages.value[messages.value.length - 1].content = `❌ ${String(e)}`
        streaming.value = false
      })
  }

  function cancelStream() {
    controller.value?.abort()
    streaming.value = false
  }

  return {
    sessions,
    currentId,
    messages,
    streaming,
    cost,
    loadSessions,
    loadCost,
    select,
    create,
    remove,
    rename,
    setMode,
    uploadFiles,
    send,
    cancelStream,
  }
})
