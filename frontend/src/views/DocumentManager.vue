<template>
  <div>
    <h2>文档管理</h2>
    <FileUpload @upload="handleUpload" />
    <div v-if="loading" style="margin-top:20px">加载中...</div>
    <DocList v-else :docs="documents" @delete="handleDelete" />
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";
import * as api from "../api";
import FileUpload from "../components/FileUpload.vue";
import DocList from "../components/DocList.vue";

const documents = ref([]);
const loading = ref(false);

async function loadDocs() {
  loading.value = true;
  try {
    const res = await api.listDocuments();
    documents.value = (res.data.documents || []).map((t) => ({ title: t }));
  } finally {
    loading.value = false;
  }
}

async function handleUpload(file) {
  try {
    const res = await api.uploadDocument(file);
    ElMessage.success(res.data.message);
    await loadDocs();
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || "上传失败");
  }
}

async function handleDelete(title) {
  try {
    await ElMessageBox.confirm(`确认删除 "${title}"？`, "警告", { type: "warning" });
    await api.deleteDocument(title);
    ElMessage.success("删除成功");
    await loadDocs();
  } catch (e) { /* canceled */ }
}

onMounted(loadDocs);
</script>
