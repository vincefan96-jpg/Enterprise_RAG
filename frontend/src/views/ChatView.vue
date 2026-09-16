<template>
  <div class="chat-layout">
    <aside class="sessions">
      <el-button
        class="new-session"
        type="primary"
        plain
        size="small"
        :disabled="streaming"
        @click="onNewSession"
      >
        ＋ 新建对话
      </el-button>
      <div class="session-list">
        <div
          v-for="s in sessions"
          :key="s.id"
          :class="['session', { active: s.id === state.activeId }]"
          @click="onSelect(s.id)"
        >
          <div class="session-main">
            <div class="session-title">{{ s.title }}</div>
            <div class="session-meta">
              {{ formatTime(s.updatedAt) }} · {{ s.messages.length }} 条
            </div>
          </div>
          <el-button
            class="session-del"
            link
            type="danger"
            size="small"
            @click.stop="onDelete(s.id)"
          >
            删除
          </el-button>
        </div>
      </div>
    </aside>

    <div class="chat-container">
      <div class="messages" ref="msgContainer">
        <template v-for="(m, i) in visibleMessages" :key="i">
          <ChatMessage v-bind="m" />
          <AgentTrace v-if="m.steps && m.steps.length" :steps="m.steps" />
        </template>
        <template v-if="streaming">
          <ChatMessage
            v-if="streamText || streamStatus"
            role="assistant"
            :text="streamText"
            :status="streamStatus"
          />
          <AgentTrace v-if="streamSteps.length" :steps="streamSteps" />
        </template>
      </div>
      <div class="input-area">
        <el-button size="small" color="#f56c6c" v-if="streaming" @click="cancelStream">停止生成</el-button>
        <el-button size="small" @click="onClearHistory" :disabled="streaming || !visibleMessages.length">清空对话</el-button>
        <el-input v-model="question" @keyup.enter="send" placeholder="输入问题..." :disabled="streaming" />
        <el-button type="primary" @click="send" :disabled="streaming || !question">发送</el-button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, onUnmounted, ref, watch } from "vue";
import { ElMessage } from "element-plus";
import * as api from "../api";
import ChatMessage from "../components/ChatMessage.vue";
import AgentTrace from "../components/AgentTrace.vue";
import { useSessions } from "../composables/useSessions";

const {
  state,
  sessions,
  activeSession,
  createSession,
  selectSession,
  deleteSession,
  touch,
  clearMessages,
} = useSessions();

const messages = computed(() => activeSession.value?.messages || []);
const visibleMessages = computed(() => {
  const restored = activeSession.value?.partial;
  if (!restored || streaming.value) return messages.value;
  const list = messages.value;
  const last = list[list.length - 1];
  const base =
    last && last.role === "user" && last.text === restored.question
      ? list.slice(0, -1)
      : list;
  return [
    ...base,
    { role: "user", text: restored.question, sources: [] },
    {
      role: "assistant",
      text: restored.answer,
      sources: restored.sources || [],
      interrupted: true,
    },
  ];
});

const question = ref("");
const streaming = ref(false);
const streamText = ref("");
const streamStatus = ref("");
const streamSources = ref([]);
const streamSteps = ref([]);
const streamError = ref("");
const msgContainer = ref(null);
const abortController = ref(null);

watch(
  () => activeSession.value?.id,
  () => scrollToBottom(),
);

function formatTime(ts) {
  if (!ts) return "";
  const date = new Date(ts);
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function onNewSession() {
  createSession();
  scrollToBottom();
}

function onSelect(id) {
  if (!streaming.value) selectSession(id);
}

function onDelete(id) {
  if (streaming.value) return;
  deleteSession(id);
}

function onClearHistory() {
  const session = activeSession.value;
  if (session) clearMessages(session.id);
}

function cancelStream() {
  if (abortController.value) abortController.value.abort();
}

function scrollToBottom() {
  nextTick(() => {
    if (msgContainer.value) {
      msgContainer.value.scrollTop = msgContainer.value.scrollHeight;
    }
  });
}

function buildHistory() {
  return messages.value
    .filter((m) => m.text)
    .slice(-12)
    .map((m) => ({ role: m.role, content: m.text }));
}

onUnmounted(() => {
  if (abortController.value) abortController.value.abort();
});

async function send() {
  const q = question.value.trim();
  const session = activeSession.value;
  if (!q || !session) return;

  const history = buildHistory();
  touch(session, q);
  session.messages.push({ role: "user", text: q, sources: [] });
  session.partial = null;
  question.value = "";
  streaming.value = true;
  streamText.value = "";
  streamStatus.value = "";
  streamSources.value = [];
  streamSteps.value = [];
  streamError.value = "";

  if (abortController.value) abortController.value.abort();
  abortController.value = new AbortController();

  const savePartial = (answer, sources) => {
    session.partial = { question: q, answer, sources };
  };

  try {
    let answer = "";
    let finalSources = [];
    await api.streamQuery(
      { question: q, history, session_id: session.threadId },
      {
        onContent: (chunk) => {
          answer += chunk;
          streamText.value = answer;
          streamStatus.value = "";
          savePartial(answer, finalSources);
          scrollToBottom();
        },
        onStatus: (status) => {
          streamStatus.value = status;
          scrollToBottom();
        },
        onSources: (sources) => {
          finalSources = sources;
          streamSources.value = sources;
        },
        onStep: (step) => {
          streamSteps.value.push(step);
          scrollToBottom();
        },
        onError: (message) => {
          streamError.value = message;
          streamStatus.value = "⚠️ " + message;
          scrollToBottom();
        },
      },
      abortController.value.signal,
    );
    if (answer || finalSources.length) {
      session.messages.push({
        role: "assistant",
        text: answer,
        sources: finalSources,
        steps: streamSteps.value,
      });
      session.partial = null;
    } else if (streamError.value) {
      ElMessage.error(streamError.value);
    }
  } catch (e) {
    if (streamText.value) {
      session.messages.push({
        role: "assistant",
        text: streamText.value,
        sources: streamSources.value,
        steps: streamSteps.value,
        interrupted: true,
      });
      session.partial = null;
    } else if (e.name !== "AbortError") {
      ElMessage.error("查询失败");
    }
  } finally {
    streaming.value = false;
    streamText.value = "";
    streamStatus.value = "";
    streamSources.value = [];
    streamSteps.value = [];
    streamError.value = "";
    abortController.value = null;
    touch(session);
  }
}
</script>

<style scoped>
.chat-layout { display: flex; gap: 16px; height: calc(100vh - 120px); }
.sessions { width: 260px; flex-shrink: 0; display: flex; flex-direction: column; gap: 8px; border-right: 1px solid #eee; padding-right: 12px; }
.new-session { width: 100%; }
.session-list { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 4px; }
.session { display: flex; align-items: center; gap: 4px; padding: 8px 10px; border-radius: 6px; cursor: pointer; }
.session:hover { background: #f5f7fa; }
.session.active { background: #ecf5ff; }
.session-main { flex: 1; min-width: 0; }
.session-title { font-size: 13px; color: #303133; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.session-meta { font-size: 11px; color: #909399; margin-top: 2px; }
.session-del { visibility: hidden; }
.session:hover .session-del { visibility: visible; }
.chat-container { flex: 1; min-width: 0; display: flex; flex-direction: column; }
.messages { flex: 1; overflow-y: auto; padding: 16px; }
.input-area { display: flex; gap: 8px; padding: 16px; border-top: 1px solid #eee; }
</style>
