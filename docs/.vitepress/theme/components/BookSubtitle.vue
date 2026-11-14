<script setup lang="ts">
import { books } from '../../sidebar.json'

// 一本书落地页标题下面的英文书名，和首页卡片读同一个 subtitle（见 scripts/build-site.mjs）
const props = defineProps<{ book: string }>()
const subtitle = books[props.book as keyof typeof books].subtitle
</script>

<template>
  <blockquote><p>{{ subtitle }}</p></blockquote>
</template>
