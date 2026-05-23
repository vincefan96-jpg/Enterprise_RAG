<template>
  <div :class="['message', role]">
    <div class="avatar">{{ role === "user" ? "我" : "AI" }}</div>
    <div class="content">
      <div v-if="status && !text" class="text thinking">{{ status }}<span class="dots">...</span></div>
      <div v-else class="text">{{ text }}<span v-if="interrupted" class="interrupted-tag">（回答被中断）</span></div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  role: String,
  text: String,
  interrupted: { type: Boolean, default: false },
  status: { type: String, default: "" },
});
</script>

<style scoped>
.message { display: flex; gap: 12px; margin-bottom: 16px; }
.message.user { flex-direction: row-reverse; }
.avatar { width: 32px; height: 32px; border-radius: 50%; background: #409eff; color: #fff; display: flex; align-items: center; justify-content: center; font-size: 12px; flex-shrink: 0; }
.message.user .avatar { background: #67c23a; }
.content { max-width: 70%; }
.text { background: #f0f0f0; padding: 12px; border-radius: 8px; white-space: pre-wrap; }
.message.user .text { background: #409eff; color: #fff; }
.interrupted-tag { color: #e6a23c; font-size: 12px; }
.thinking { color: #909399; }
.dots { animation: pulse 1.5s ease-in-out infinite; }
@keyframes pulse {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 1; }
}
</style>
