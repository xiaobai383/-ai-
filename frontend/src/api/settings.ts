import { http } from './request'

export interface Settings {
  model_name: string
  model_base_url: string
  api_key_masked: string
  fallback_enabled: boolean
  fallback_ollama_base_url: string
  fallback_ollama_model: string
}

export interface SettingsUpdate {
  model_name?: string
  model_base_url?: string
  api_key?: string
  fallback_enabled?: boolean
  fallback_ollama_base_url?: string
  fallback_ollama_model?: string
}

export async function getSettings(): Promise<Settings> {
  const { data } = await http.get('/settings')
  return data as Settings
}

export async function updateSettings(payload: SettingsUpdate): Promise<Settings> {
  const { data } = await http.patch('/settings', payload)
  return data as Settings
}
