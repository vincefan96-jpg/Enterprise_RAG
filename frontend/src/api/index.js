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

export function streamQuery(question, onContent, onStatus, onSources, signal) {
  return fetch("/api/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  }).then(async (response) => {
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
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data === "[DONE]") return;
          try {
            const json = JSON.parse(data);
            if (json.type === "content" && json.content) onContent(json.content);
            else if (json.type === "status" && json.message) onStatus(json.message);
            else if (json.type === "sources" && json.sources) onSources(json.sources);
            else if (json.type === "error" && json.message) onStatus("⚠️ " + json.message);
          } catch (e) { /* skip unparseable lines */ }
        }
      }
    }
  });
}
