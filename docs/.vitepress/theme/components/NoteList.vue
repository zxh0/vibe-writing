<script setup lang="ts">
import { withBase } from 'vitepress'
import { notes } from '../../sidebar.json'
</script>

<template>
  <ul class="note-list">
    <li v-for="n in notes.latest" :key="n.link">
      <span class="date">{{ n.date }}</span>
      <a :href="withBase(n.link)">{{ n.title }}</a>
    </li>
  </ul>
</template>
