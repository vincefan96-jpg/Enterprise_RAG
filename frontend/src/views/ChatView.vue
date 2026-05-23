<template>
  <div class="chat-container">
    <div class="messages" ref="msgContainer">
      <ChatMessage v-for="(m, i) in messages" :key="i" v-bind="m" />
      <ChatMessage
        v-if="streaming"
        role="assistant"
        :text="streamText"
        :status="streamStatus"
      />
    </div>
    <div class="input-area">
      <el-button size="small" color="#f56c6c" v-if="streaming" @click="cancelStream">停止生成</el-button>
      <el-button size="small" @click="clearHistory" :disabled="streaming || messages.length === 0">清空对话</el-button>
      <el-input v-model="question" @keyup.enter="send" placeholder="输入问题..." :disabled="streaming" />
      <el-button type="primary" @click="send" :disabled="streaming || !question">发送</el-button>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick, onUnmounted } from "vue";
import { ElMessage } from "element-plus";
import * as api from "../api";
import ChatMessage from "../components/ChatMessage.vue";

const STORAGE_KEY = "rag_chat_messages";
const STREAM_KEY = "rag_chat_streaming";
const SAVE_THROTTLE_MS = 500;

function loadMessages() {
  const msgs = [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) msgs.push(...parsed);
    }
  } catch { /* corrupted data */ }

  const partial = loadPartial();
  if (partial) {
    msgs.push({ role: "user", text: partial.question, sources: [] });
    msgs.push({ role: "assistant", text: partial.answer, sources: partial.sources || [], interrupted: true });
  }
  return msgs;
}

function loadPartial() {
  try {
    const raw = localStorage.getItem(STREAM_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed.question && parsed.answer) return parsed;
    }
  } catch { /* corrupted */ }
  return null;
}

let _lastSave = 0;
function savePartial(question, answer, sources) {
  const now = Date.now();
  if (now - _lastSave < SAVE_THROTTLE_MS) return;
  _lastSave = now;
  localStorage.setItem(STREAM_KEY, JSON.stringify({ question, answer, sources }));
}

function clearPartial() {
  localStorage.removeItem(STREAM_KEY);
}

const messages = ref(loadMessages());
const question = ref("");
const streaming = ref(false);
const streamText = ref("");
const streamStatus = ref("");
const streamSources = ref([]);
const msgContainer = ref(null);
const abortController = ref(null);

watch(messages, (val) => {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(val));
}, { deep: true });

function clearHistory() {
  messages.value = [];
  localStorage.removeItem(STORAGE_KEY);
  clearPartial();
}

function cancelStream() {
  if (abortController.value) {
    abortController.value.abort();
  }
}

onUnmounted(() => {
  if (abortController.value) {
    abortController.value.abort();
  }
});

async function send() {
  const q = question.value.trim();
  if (!q) return;
  messages.value.push({ role: "user", text: q, sources: [] });
  question.value = "";
  streaming.value = true;
  streamText.value = "";
  streamStatus.value = "";
  streamSources.value = [];
  clearPartial();
  _lastSave = 0;

  if (abortController.value) {
    abortController.value.abort();
  }
  abortController.value = new AbortController();

  try {
    let answer = "";
    let finalSources = [];
    await api.streamQuery(
      q,
      (chunk) => {
        answer += chunk;
        streamText.value = answer;
        streamStatus.value = "";
        savePartial(q, answer, finalSources);
        nextTick(() => { msgContainer.value.scrollTop = msgContainer.value.scrollHeight; });
      },
      (status) => {
        streamStatus.value = status;
        nextTick(() => { msgContainer.value.scrollTop = msgContainer.value.scrollHeight; });
      },
      (sources) => {
        finalSources = sources;
        streamSources.value = sources;
      },
      abortController.value.signal,
    );
    messages.value.push({ role: "assistant", text: answer, sources: finalSources });
  } catch (e) {
    if (e.name === "AbortError") {
      if (streamText.value) {
        messages.value.push({ role: "assistant", text: streamText.value, sources: streamSources.value, interrupted: true });
      }
      return;
    }
    if (streamText.value) {
      messages.value.push({ role: "assistant", text: streamText.value, sources: streamSources.value, interrupted: true });
    } else {
      ElMessage.error("查询失败");
    }
  } finally {
    streaming.value = false;
    streamText.value = "";
    streamStatus.value = "";
    streamSources.value = [];
    clearPartial();
    abortController.value = null;
  }
}
</script>

<style scoped>
.chat-container { display: flex; flex-direction: column; height: calc(100vh - 120px); }
.messages { flex: 1; overflow-y: auto; padding: 16px; }
.input-area { display: flex; gap: 8px; padding: 16px; border-top: 1px solid #eee; }
</style>
