<template>
  <el-table :data="docs" style="width:100%">
    <el-table-column prop="title" label="文档名称" />
    <el-table-column label="操作" width="120">
      <template #default="scope">
        <el-button type="danger" size="small" @click="$emit('delete', scope.row.title)">删除</el-button>
      </template>
    </el-table-column>
  </el-table>
</template>

<script setup>
defineProps({ docs: Array });
defineEmits(["delete"]);
</script>
