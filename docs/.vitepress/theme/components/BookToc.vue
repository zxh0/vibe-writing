<script setup lang="ts">
import { computed } from 'vue'
import { withBase } from 'vitepress'
import { books } from '../../sidebar.json'

// 一本书的目录：已发布的章节能点开，还没写完的先灰着列出来。
// from/to 是 toc 数组的下标区间（左闭右开），用来把目录分段展示。
const props = defineProps<{ book: string; from?: number; to?: number }>()
const items = computed(() => {
  const toc = books[props.book as keyof typeof books].toc
  return toc.slice(props.from ?? 0, props.to ?? toc.length)
})
</script>

<template>
  <ul class="book-toc">
    <li v-for="t in items" :key="t.text" :class="{ published: t.link }">
      <a v-if="t.link" :href="withBase(t.link)">{{ t.text }}</a>
      <template v-else>{{ t.text }}</template>
    </li>
  </ul>
</template>
