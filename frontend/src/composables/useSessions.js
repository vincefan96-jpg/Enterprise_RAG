import { computed, ref, watch } from "vue";

const STORAGE_KEY = "rag_sessions_v1";
const LEGACY_KEYS = ["rag_chat_messages", "rag_chat_streaming", "rag_chat_session"];
const SAVE_THROTTLE_MS = 400;
const DEFAULT_TITLE = "新对话";

function uuid() {
  return crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function newSession() {
  const now = Date.now();
  return {
    id: uuid(),
    threadId: uuid(),
    title: DEFAULT_TITLE,
    createdAt: now,
    updatedAt: now,
    messages: [],
    partial: null,
  };
}

function migrateLegacy() {
  let messages = [];
  let partial = null;
  let threadId = null;
  try {
    const raw = localStorage.getItem("rag_chat_messages");
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) messages = parsed;
    }
    const partialRaw = localStorage.getItem("rag_chat_streaming");
    if (partialRaw) partial = JSON.parse(partialRaw);
    threadId = localStorage.getItem("rag_chat_session");
  } catch {
    /* corrupted legacy data */
  }
  LEGACY_KEYS.forEach((key) => localStorage.removeItem(key));

  if (!messages.length && !partial) return null;

  const session = newSession();
  session.messages = messages;
  session.partial = partial;
  if (threadId) session.threadId = threadId;
  return session;
}

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const data = JSON.parse(raw);
      if (data && Array.isArray(data.sessions) && data.sessions.length) {
        return { activeId: data.activeId || data.sessions[0].id, sessions: data.sessions };
      }
    }
  } catch {
    /* corrupted storage */
  }

  const legacy = migrateLegacy();
  const session = legacy || newSession();
  return { activeId: session.id, sessions: [session] };
}

const state = ref(loadState());

let lastSave = 0;
let saveTimer = null;

watch(
  state,
  () => {
    const persist = () => {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state.value));
        lastSave = Date.now();
      } catch {
        /* storage disabled */
      }
    };
    const elapsed = Date.now() - lastSave;
    if (elapsed >= SAVE_THROTTLE_MS) {
      persist();
    } else {
      clearTimeout(saveTimer);
      saveTimer = setTimeout(persist, SAVE_THROTTLE_MS - elapsed);
    }
  },
  { deep: true },
);

const sessions = computed(() =>
  [...state.value.sessions].sort((a, b) => b.updatedAt - a.updatedAt),
);

const activeSession = computed(
  () => state.value.sessions.find((s) => s.id === state.value.activeId) || null,
);

function createSession() {
  const session = newSession();
  state.value.sessions.push(session);
  state.value.activeId = session.id;
  return session;
}

function selectSession(id) {
  if (state.value.sessions.some((s) => s.id === id)) {
    state.value.activeId = id;
  }
}

function deleteSession(id) {
  const index = state.value.sessions.findIndex((s) => s.id === id);
  if (index < 0) return;

  state.value.sessions.splice(index, 1);
  if (!state.value.sessions.length) {
    state.value.sessions.push(newSession());
  }
  if (state.value.activeId === id) {
    const fallback = sessions.value[0];
    state.value.activeId = fallback.id;
  }
}

function touch(session, title) {
  if (!session) return;
  session.updatedAt = Date.now();
  if (title && (!session.title || session.title === DEFAULT_TITLE)) {
    session.title = title.length > 24 ? `${title.slice(0, 24)}…` : title;
  }
}

function clearMessages(id) {
  const session = state.value.sessions.find((s) => s.id === id);
  if (!session) return;
  session.messages = [];
  session.partial = null;
  session.threadId = uuid();
  session.updatedAt = Date.now();
}

export function useSessions() {
  return {
    state,
    sessions,
    activeSession,
    createSession,
    selectSession,
    deleteSession,
    touch,
    clearMessages,
  };
}
