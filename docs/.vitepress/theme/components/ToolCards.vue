<script setup lang="ts">
import { withBase } from 'vitepress'
import { books } from '../../sidebar.json'

const props = defineProps<{ book: string }>()
const tools = books[props.book as keyof typeof books].tools
</script>

<template>
  <div class="tool-cards">
    <a v-for="t in tools" :key="t.link" class="tool-card" :href="withBase(t.link)" target="_blank" rel="noopener">
      <h3>{{ t.title }}</h3>
      <p class="tool-card-blurb">{{ t.blurb }}</p>
      <p class="tool-card-open">打开工具 →</p>
    </a>
  </div>
</template>
