import { createRouter, createWebHistory } from 'vue-router'
import ChatView from '../views/ChatView.vue'
import SettingsView from '../views/SettingsView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'chat', component: ChatView },
    { path: '/settings', name: 'settings', component: SettingsView },
    { path: '/task', redirect: '/' },
    { path: '/chat', redirect: '/' },
  ],
})

export default router
