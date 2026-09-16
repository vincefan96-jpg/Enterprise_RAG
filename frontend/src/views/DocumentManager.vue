<template>
  <div>
    <h2>文档管理</h2>
    <FileUpload @upload="handleUpload" :disabled="uploading" />
    <div v-if="uploading" style="margin-top:12px;color:#909399">
      上传处理中，请稍候（解析与向量化需要一些时间）...
    </div>
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
const uploading = ref(false);

async function loadDocs() {
  loading.value = true;
  try {
    const res = await api.listDocuments();
    documents.value = (res.data.documents || []).map((t) => ({ title: t }));
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || "文档列表加载失败，请检查后端/Milvus 是否正常");
  } finally {
    loading.value = false;
  }
}

async function handleUpload(file) {
  uploading.value = true;
  try {
    const res = await api.uploadDocument(file);
    ElMessage.success(res.data.message);
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || "上传失败");
  } finally {
    uploading.value = false;
    await loadDocs();
  }
}

async function handleDelete(title) {
  try {
    await ElMessageBox.confirm(`确认删除 "${title}"？`, "警告", { type: "warning" });
  } catch {
    return;
  }
  try {
    await api.deleteDocument(title);
    ElMessage.success("删除成功");
    await loadDocs();
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || "删除失败");
  }
}

onMounted(loadDocs);
</script>
