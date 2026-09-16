import axios from "axios";

const api = axios.create({ baseURL: "/api" });

export function uploadDocument(file) {
  const form = new FormData();
  form.append("file", file);
  return api.post("/documents/upload", form);
}

export function listDocuments() {
  return api.get("/documents/list");
}

export function deleteDocument(title) {
  return api.delete(`/documents/${encodeURIComponent(title)}`);
}

export function queryRAG(question) {
  return api.post("/query", { question });
}

export function streamQuery(payload, handlers, signal) {
  const { onContent, onStatus, onSources, onStep, onError } = handlers || {};
  return fetch("/api/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  }).then(async (response) => {
    if (!response.ok) {
      throw new Error(`请求失败: ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6);
        if (data === "[DONE]") return;
        try {
          const json = JSON.parse(data);
          if (json.type === "content" && json.content && onContent) onContent(json.content);
          else if (json.type === "status" && json.message && onStatus) onStatus(json.message);
          else if (json.type === "sources" && json.sources && onSources) onSources(json.sources);
          else if (json.type === "step" && onStep) onStep({ node: json.node, detail: json.detail });
          else if (json.type === "error" && json.message && onError) onError(json.message);
        } catch (e) { /* skip unparseable lines */ }
      }
    }
  });
}
