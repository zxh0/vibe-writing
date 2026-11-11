import { execFileSync } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vitepress'
import { notes, books, rewrites } from './sidebar.json'

// sidebar.json 由 scripts/build-site.mjs 从 llm_notes/ 和各本书生成
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const repo = 'https://github.com/zxh0/vibe-writing'
const bookList = Object.values(books)

// 生成的页面文件沿用仓库里的路径（docs/ann_book/chapters/03_ch01_neuron.md），
// 这个正则认出它们，editLink 和 lastUpdated 都靠它对回仓库里的源文件。
const SOURCE_RE = /^(llm_notes|[a-z_]+_book)\/(chapters\/|\d{4}-\d{2}-\d{2}-)/

export default defineConfig({
  title: '学 AI，从零开始',
  description: '一个「图解 LLM」笔记系列，和几本还在写的书：人工神经网络、LLM 推理引擎、训练 LLM、智能体、GPU、TPU。',
  lang: 'zh-CN',
  base: '/vibe-writing/',
  cleanUrls: true,
  lastUpdated: true,
  rewrites,
  head: [
    ['meta', { property: 'og:title', content: '学 AI，从零开始' }],
    ['meta', { property: 'og:image', content: 'https://zxh0.github.io/vibe-writing/ann/cover.png' }],
  ],

  // 生成的页面不在 git 里，VitePress 自带的 lastUpdated 查不到提交时间；
  // 这里对回仓库里的源文件再查一次。
  transformPageData(pageData) {
    const m = pageData.filePath.match(SOURCE_RE)
    if (!m) return
    try {
      const out = execFileSync('git', ['log', '-1', '--format=%at', '--', pageData.filePath], {
        cwd: root,
        encoding: 'utf8',
      }).trim()
      if (out) pageData.lastUpdated = Number(out) * 1000
    } catch {
      // 没有 git 历史（比如浅克隆）就不显示
    }
  },

  markdown: {
    lineNumbers: false,
    image: { lazyLoading: true },
    // 正文里有 $...$ 和 $$...$$ 公式，需要 markdown-it-mathjax3
    math: true,
    // 插图的 width（像素宽度的一半）由 scripts/build-site.mjs 放在图片的 title 位
    // 带过来：![alt](url "width=N")。这里把它还原成 <img width="N">，并去掉 title，
    // 免得鼠标悬停时冒出一个 "width=N" 的提示。
    config(md) {
      const image = md.renderer.rules.image!
      md.renderer.rules.image = (tokens, idx, options, env, self) => {
        const token = tokens[idx]
        const m = /^width=(\d+)$/.exec(token.attrGet('title') ?? '')
        if (m) {
          token.attrs = token.attrs!.filter(([name]) => name !== 'title')
          token.attrSet('width', m[1])
        }
        return image(tokens, idx, options, env, self)
      }
    },
  },

  themeConfig: {
    outline: { level: [2, 3], label: '本页目录' },

    nav: [
      { text: '笔记', link: notes.link, activeMatch: `^${notes.link}` },
      {
        text: '书',
        items: bookList.map((b) => ({ text: b.title, link: b.link, activeMatch: `^${b.link}` })),
      },
      { text: '提问纠错', link: `${repo}/issues` },
    ],

    // 每个区一套侧边栏：笔记按时间正序；书是「进度和目录」+ 已上站的章节 + 小工具
    sidebar: {
      [notes.link]: [{ text: notes.title, link: notes.link, items: notes.sidebar }],
      ...Object.fromEntries(
        bookList.map((b) => [
          b.link,
          [
            {
              text: b.title,
              items: [
                { text: '进度和目录', link: b.link },
                ...b.sidebar,
                ...(b.tools.length ? [{ text: '小工具', link: `${b.link}/tools` }] : []),
              ],
            },
          ],
        ])
      ),
    },

    socialLinks: [{ icon: 'github', link: repo }],

    editLink: {
      // 生成的章节和笔记页面，filePath 就是仓库里的源文件路径；
      // docs/ 下手写的落地页对回 docs/；其余（生成的工具列表页）没有源文件，指到 issues。
      // 注意：这个函数和下面的分词器一样，会被序列化后送到浏览器执行，必须自包含，
      // 不能引用 sidebar.json 或上面的变量。
      pattern: ({ filePath }) => {
        const repo = 'https://github.com/zxh0/vibe-writing'
        if (/^(llm_notes|[a-z_]+_book)\/(chapters\/|\d{4}-\d{2}-\d{2}-)/.test(filePath))
          return `${repo}/edit/main/${filePath}`
        if (!filePath.includes('/')) return `${repo}/edit/main/docs/${filePath}`
        return `${repo}/issues`
      },
      text: '在 GitHub 上纠错',
    },

    search: {
      provider: 'local',
      options: {
        // 默认分词器按空白切词，对中文几乎无效。这里把中文按字拆开建索引，
        // 配合 AND 组合，搜「注意力」= 同时包含这三个字的页面。
        // 注意：这个函数会被序列化后送到浏览器执行，必须自包含。
        miniSearch: {
          options: {
            tokenize: (text: string) =>
              text
                .split(/[^\p{L}\p{N}_]+/u)
                .flatMap((w) => (/[一-鿿]/.test(w) ? w.split('') : [w]))
                .filter(Boolean),
            processTerm: (term: string) => term.toLowerCase(),
          },
          searchOptions: {
            combineWith: 'AND',
            prefix: true,
            boost: { title: 4, text: 2, titles: 1 },
          },
        },
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
          modal: {
            displayDetails: '显示详情',
            resetButtonTitle: '清除',
            noResultsText: '没有找到',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
          },
        },
      },
    },

    docFooter: { prev: '上一篇', next: '下一篇' },
    darkModeSwitchLabel: '主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    sidebarMenuLabel: '目录',
    returnToTopLabel: '回到顶部',
    lastUpdated: { text: '最后更新' },
    outlineTitle: '本页目录',

    footer: {
      message: '文字和插图均为原创，转载请注明出处',
      copyright: 'Copyright © 2026 zxh0',
    },
  },
})
