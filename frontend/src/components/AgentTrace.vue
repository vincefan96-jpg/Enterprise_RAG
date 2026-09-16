<template>
  <div v-if="normalized.length" class="agent-trace">
    <div class="trace-title">Agent 轨迹</div>
    <div v-for="(s, i) in normalized" :key="i" class="trace-item">
      <span class="node">{{ label(s.node) }}</span>
      <span class="detail">{{ s.detail }}</span>
    </div>
  </div>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  steps: { type: Array, default: () => [] },
});

const LABELS = {
  route: "意图路由",
  rewrite: "查询改写",
  decompose: "子问题分解",
  tool: "工具调用",
  retrieve: "检索命中",
  grade: "充分性评估",
  refine: "查询纠错",
  generate: "生成回答",
};

function label(node) {
  return LABELS[node] || node;
}

const DETAIL_LABELS = {
  sufficient: "充分",
  insufficient: "不足",
};

const normalized = computed(() =>
  props.steps.map((s) => ({
    node: s.node,
    detail: Array.isArray(s.detail)
      ? (s.detail.length ? s.detail.join("、") : "未命中")
      : (DETAIL_LABELS[s.detail] || String(s.detail ?? "")),
  }))
);
</script>

<style scoped>
.agent-trace {
  margin: 4px 0 12px 44px;
  padding: 8px 12px;
  border-left: 3px solid #c6e2ff;
  background: #f7fbff;
  border-radius: 6px;
  font-size: 12px;
  color: #606266;
}
.trace-title {
  font-weight: 600;
  color: #409eff;
  margin-bottom: 4px;
}
.trace-item {
  display: flex;
  gap: 8px;
  line-height: 1.8;
}
.trace-item .node {
  flex-shrink: 0;
  color: #909399;
}
.trace-item .detail {
  word-break: break-all;
}
</style>
