// 任务相关类型
export type TaskMode = 'privacy_enhanced' | 'local_fallback'
export type OutputFormat = 'markdown' | 'plain' | 'json' | 'html'

// SSE 事件协议（与后端 src/api/app.py 一致）
export interface StepEvent {
  type: 'step'
  step_id: number
  name: string
  status: 'success' | 'failed'
  duration_ms: number
  input_preview: string
  output_preview: string
  tokens_in: number
  tokens_out: number
  cost_yuan: number
}

export interface TokenEvent {
  type: 'token'
  text: string
}

export interface TaskDoneEvent {
  type: 'done'
  run_id: string
  result_path: string | null
  result_preview: string | null
  tokens_in: number
  tokens_out: number
  cost_yuan: number
  fallback: boolean
  steps: Record<string, unknown>[]
}

export interface ErrorEvent {
  type: 'error'
  message: string
}

export type SSEEvent = StepEvent | TokenEvent | TaskDoneEvent | ErrorEvent

// 对话 SSE 事件（含思考步骤 step）
export interface ChatDoneEvent {
  type: 'done'
  reply: string
  used_fallback: boolean
  tokens_in: number
  tokens_out: number
}
export type ChatSSEEvent = StepEvent | TokenEvent | ChatDoneEvent | ErrorEvent

// 会话/消息
export interface Session {
  session_id: string
  title: string
  mode?: string
  created_at?: string
  updated_at?: string
}

export interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp?: string
  steps?: StepEvent[]
}
