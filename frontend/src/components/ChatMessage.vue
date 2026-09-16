<template>
  <div :class="['message', role]">
    <div class="avatar">{{ role === "user" ? "我" : "AI" }}</div>
    <div class="content">
      <div v-if="status && !text" class="text thinking">{{ status }}<span class="dots">...</span></div>
      <div v-else class="text">{{ displayText }}<span v-if="interrupted" class="interrupted-tag">（回答被中断）</span></div>
      <div v-if="sources && sources.length" class="sources">
        <span class="sources-label">来源：</span>
        <span v-for="(s, i) in sources" :key="i" class="source">{{ s }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  role: String,
  text: String,
  interrupted: { type: Boolean, default: false },
  status: { type: String, default: "" },
  sources: { type: Array, default: () => [] },
});

// Backend keeps [n] citation markers in the answer for source attribution;
// hide them in the UI and keep only the source tags below.
const displayText = computed(() =>
  (props.text || "")
    .replace(/\s*\[\d+\]/g, "")
    .replace(/\s*\[\d+$/, "")
);
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
.sources { margin-top: 6px; font-size: 12px; color: #909399; }
.sources-label { margin-right: 4px; }
.source { display: inline-block; background: #ecf5ff; color: #409eff; border-radius: 4px; padding: 1px 6px; margin: 2px 4px 0 0; }
@keyframes pulse {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 1; }
}
</style>
