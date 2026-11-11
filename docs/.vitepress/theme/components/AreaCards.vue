<script setup lang="ts">
import { withBase } from 'vitepress'
import { notes, books } from '../../sidebar.json'

// 首页的卡片：笔记排第一，后面是各本书，顺序跟 scripts/build-site.mjs 里的 BOOKS 表一致
const cards = [
  {
    ...notes,
    cover: undefined as string | undefined,
    progress: `共 ${notes.count} 篇，最新一篇 ${notes.latest[0].date}`,
  },
  ...Object.values(books).map((b) => ({
    ...b,
    progress:
      b.sidebar.length > 0
        ? `目录 ${b.toc.length} 章，已发布 ${b.sidebar.length} 章`
        : `目录 ${b.toc.length} 章，还没有发布的章节`,
  })),
]
</script>

<template>
  <div class="area-cards">
    <a v-for="c in cards" :key="c.link" class="area-card" :href="withBase(c.link)">
      <img v-if="c.cover" :src="withBase(c.cover)" :alt="c.title" />
      <div v-else class="area-card-placeholder">{{ c.subtitle }}</div>
      <div class="area-card-body">
        <h3>{{ c.title }}</h3>
        <p class="area-card-sub">{{ c.subtitle }} · <span class="area-card-status">{{ c.status }}</span></p>
        <p class="area-card-blurb">{{ c.blurb }}</p>
        <p class="area-card-progress">{{ c.progress }}</p>
      </div>
    </a>
  </div>
</template>
