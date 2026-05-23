import { createRouter, createWebHistory } from "vue-router";

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/chat" },
    { path: "/documents", component: () => import("../views/DocumentManager.vue") },
    { path: "/chat", component: () => import("../views/ChatView.vue") },
  ],
});
