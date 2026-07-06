import { http, streamSSE } from './request'
import type { Session, Message, ChatSSEEvent } from './types'

export async function listSessions(): Promise<Session[]> {
  const { data } = await http.get('/sessions')
  return data.sessions as Session[]
}

export async function createSession(title = '新建会话', mode = 'privacy_enhanced'): Promise<Session> {
  const { data } = await http.post('/sessions', { title, mode })
  return data as Session
}

export async function deleteSession(id: string): Promise<void> {
  await http.delete(`/sessions/${id}`)
}

export async function renameSession(id: string, title: string): Promise<Session> {
  const { data } = await http.patch(`/sessions/${id}`, { title })
  return data as Session
}

export async function updateSessionMode(id: string, mode: string): Promise<Session> {
  const { data } = await http.patch(`/sessions/${id}`, { mode })
  return data as Session
}

export async function getMessages(id: string): Promise<Message[]> {
  const { data } = await http.get(`/sessions/${id}/messages`)
  return data.messages as Message[]
}

export async function uploadSessionFiles(id: string, files: File[]): Promise<number> {
  const form = new FormData()
  files.forEach((f) => form.append('files', f))
  const { data } = await http.post(`/sessions/${id}/upload`, form)
  return data.chunks_indexed as number
}

export async function getSessionCost(id: string): Promise<{
  session_id: string
  tokens_in: number
  tokens_out: number
  cost_yuan: number
  balance_yuan: number | null
  budget_yuan: number
}> {
  const { data } = await http.get(`/sessions/${id}/cost`)
  return data
}

/** 流式发送消息：SSE 推送 token/done */
export async function sendMessageStream(
  sessionId: string,
  message: string,
  onEvent: (evt: ChatSSEEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  return streamSSE<ChatSSEEvent>(
    `/sessions/${sessionId}/messages/stream`,
    JSON.stringify({ message }),
    onEvent,
    signal,
  )
}
