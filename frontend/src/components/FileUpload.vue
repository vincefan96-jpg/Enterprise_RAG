<template>
  <el-upload
    drag
    :auto-upload="false"
    :on-change="handleChange"
    :limit="1"
    accept=".pdf,.docx,.doc,.txt"
  >
    <el-icon><upload-filled /></el-icon>
    <div>拖拽文件到此处或点击上传</div>
    <template #tip>支持 PDF / Word / TXT，最大 50MB</template>
  </el-upload>
</template>

<script setup>
import { UploadFilled } from "@element-plus/icons-vue";
const emit = defineEmits(["upload"]);
function handleChange(file) { emit("upload", file.raw); }
</script>
