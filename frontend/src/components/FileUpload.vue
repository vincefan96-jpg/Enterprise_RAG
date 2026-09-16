<template>
  <el-upload
    ref="uploadRef"
    drag
    :auto-upload="false"
    :disabled="disabled"
    :on-change="handleChange"
    :on-exceed="handleExceed"
    :limit="1"
    accept=".pdf,.docx,.txt,.md"
  >
    <el-icon><upload-filled /></el-icon>
    <div>拖拽文件到此处或点击上传</div>
    <template #tip>支持 PDF / Word / TXT / Markdown，最大 50MB</template>
  </el-upload>
</template>

<script setup>
import { ref } from "vue";
import { genFileId } from "element-plus";
import { UploadFilled } from "@element-plus/icons-vue";

defineProps({ disabled: { type: Boolean, default: false } });

const emit = defineEmits(["upload"]);
const uploadRef = ref(null);

function handleChange(file) {
  emit("upload", file.raw);
  uploadRef.value?.clearFiles();
}

function handleExceed(files) {
  uploadRef.value?.clearFiles();
  const file = files[0];
  file.uid = genFileId();
  uploadRef.value?.handleStart(file);
}
</script>
