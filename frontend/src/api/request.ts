import axios from 'axios'

// API 前缀：
//   开发（npm run dev）：/api 由 Vite proxy 转发到后端 :8000
//   生产（npm run build 后由 FastAPI 直接 serve）：设为空，请求打同域根路径
export const API_BASE = (import.meta.env.VITE_API_BASE as string) ?? '/api'

export const http = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
})

/**
 * SSE 流式消费：POST 请求 + ReadableStream 逐帧解析。
 * 支持 JSON body（string）与 multipart body（FormData）。
 * POST 不能用 EventSource（仅 GET），故用 fetch + getReader。
 */
export async function streamSSE<T>(
  url: string,
  body: BodyInit,
  onEvent: (evt: T) => void,
  signal?: AbortSignal,
): Promise<void> {
  const headers: Record<string, string> = {}
  if (!(body instanceof FormData)) {
    headers['Content-Type'] = 'application/json'
  }
  const resp = await fetch(`${API_BASE}${url}`, {
    method: 'POST',
    headers,
    body,
    signal,
  })
  if (!resp.ok || !resp.body) {
    throw new Error(`SSE 请求失败：HTTP ${resp.status}`)
  }
  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    // SSE 帧以 \n\n 分隔
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''
    for (const frame of frames) {
      const line = frame.trim()
      if (!line.startsWith('data: ')) continue
      const json = line.slice(6)
      try {
        onEvent(JSON.parse(json) as T)
      } catch (e) {
        console.error('SSE 帧解析失败', e, json)
      }
    }
  }
}
