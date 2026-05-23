<template>
  <el-container class="app-container">
    <el-header class="app-header">
      <el-menu mode="horizontal" router :default-active="$route.path">
        <el-menu-item index="/documents">文档管理</el-menu-item>
        <el-menu-item index="/chat">知识问答</el-menu-item>
      </el-menu>
    </el-header>
    <el-main><router-view /></el-main>
  </el-container>
</template>

<style>
.app-container { min-height: 100vh; }
.app-header { padding: 0; }
</style>
