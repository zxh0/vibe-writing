import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import AreaCards from './components/AreaCards.vue'
import BookToc from './components/BookToc.vue'
import NoteList from './components/NoteList.vue'
import ToolCards from './components/ToolCards.vue'
import './custom.css'

// 首页和各区落地页用的几个小组件，数据都来自 sidebar.json：
// <AreaCards />              首页：笔记 + 各本书的卡片
// <BookToc book="ann" />     一本书的目录，已上站的章节能点开；from/to 可以只列一段
// <NoteList />               笔记列表，最新的在前
// <ToolCards book="gpu" />   一本书的小工具卡片
export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('AreaCards', AreaCards)
    app.component('BookToc', BookToc)
    app.component('NoteList', NoteList)
    app.component('ToolCards', ToolCards)
  },
} satisfies Theme
