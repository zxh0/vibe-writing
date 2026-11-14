// 从 llm_notes/ 和各本书的 chapters/、images/、aigc/、tools/ 生成 VitePress 站点内容。
// 仓库里的 Markdown 和图片是唯一的内容来源；docs/ 下除了 index.md、各区的落地页和
// .vitepress/ 里的配置和主题，其余（docs/<仓库目录>/、docs/public/、sidebar.json）都是生成物。
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const docsDir = path.join(root, 'docs')
const publicDir = path.join(docsDir, 'public')

// ---- 内容表 ----
//
// 站点地址：/notes 是笔记，/<route> 是一本书，/<route>/tools 是这本书的小工具（如果有）。
// 生成的页面文件放在 docs/<仓库目录>/ 下、沿用仓库里的相对路径（docs/ann_book/chapters/03_ch01_neuron.md），
// 再用 VitePress 的 rewrites 映射到干净的地址（/ann/ch01-neuron）。这样 editLink 拿到的
// filePath 就是仓库里的源文件路径，「在 GitHub 上纠错」能直接对回去。
//
// 表的顺序就是首页卡片和顶栏下拉的顺序。
// 书是「写完一章发一章」：往 published 里加一行文件名，那一章就上站；
// 没列进来的章节只在这本书的目录页里露个标题，正文继续待在仓库里。
//
// 英文书名（subtitle）以扉页为准：不写 subtitle 的书，取 chapters/_front.md 里书名下面
// 第一行 > 引用（跳过「版本：」那行）。首页卡片和各书落地页的 <BookSubtitle /> 都读它，
// 改书名只改扉页一处。扉页没有英文书名的（gpu、tpu、ann）才在这里直接写。

const NOTES = {
  dir: 'llm_notes',
  route: 'notes',
  title: '图解 LLM 系列笔记',
  subtitle: 'LLM Notes',
  status: '连载中',
  cover: 'aigc/front.png',
  blurb: '一篇一个主题，解读 LLM 相关的论文和核心公式，能画图的地方都画图。从 DeepSeek 的 MLA、Flash Attention、位置编码，一路写到 Kimi Delta Attention、LongCat Sparse Attention。',
}

const BOOKS = [
  {
    dir: 'toyllm_book1',
    route: 'llm-engine',
    title: '自己动手写LLM推理引擎',
    status: '连载中',
    blurb: '用 Python 和 PyTorch 从零手写一个能跑的 LLM 推理引擎，目标模型 SmolLM2-135M，跑在笔记本 CPU 上。慢，但每一个数字都能和 HuggingFace 对上。',
    cover: 'aigc/front.png',
    published: [
      '_preface.md',
      'ch01_overview.md',
      'ch02_weights.md',
      'ch03_tokenizer.md',
      'ch04_embedding.md',
      'ch05_norm.md',
      'ch06_attn.md',
      'ch07_rope.md',
      'ch08_ffn.md',
      'ch09_residual.md',
      'ch10_sampling.md',
      'ch11_kvcache.md',
      'ch12_flash_attn.md',
      'ch13_qwen3.md',
      'ch14_gpu.md',
      'x_a_matrix.md',
      'x_b_formulas.md',
      'x_c_papers.md',
      'x_d_glossary.md',
    ],
    tools: {
      'safetensors-viewer': 'safetensors 参数布局查看器',
      'sinpe-heatmap': '正弦位置编码热力图',
      'rope-angles': 'RoPE 旋转角度',
      'embed-dims': '主流模型的嵌入维度',
      'param-cubes': '参数量立方体',
    },
  },
  {
    dir: 'toyllm_book2',
    route: 'llm-train',
    title: '自己动手训练LLM',
    status: '构思和初稿',
    blurb: '《自己动手写LLM推理引擎》的姊妹篇：从零训练一个自己的小模型，最后再用第一本的引擎把它跑起来。两本书首尾相接。',
    cover: 'aigc/front.png',
    published: [],
  },
  {
    dir: 'toyllm_book3',
    route: 'agent',
    title: '自己动手写智能体',
    status: '写作计划',
    blurb: '从零写一个玩具版的 Coding Agent：一个循环，加上工具调用、权限、上下文、检索、MCP、Skills……每加一样都能跑，每章结尾对照真实的 Agent。',
    cover: 'aigc/front.png',
    published: [],
  },
  {
    dir: 'toyllm_book4',
    route: 'framework',
    title: '自己动手写深度学习框架',
    status: '写作计划',
    blurb: '只用 NumPy 从零写一个玩具版的 PyTorch：张量、自动求导、模块、优化器、数据加载，处处和真的 PyTorch 对答案，最后训出的模型交给第一本的引擎去跑。',
    published: [],
  },
  {
    dir: 'gpu_book',
    route: 'gpu',
    title: 'GPU学习笔记',
    subtitle: 'How to Make a Huge Matmul Fast',
    status: '构思和初稿',
    blurb: '整本书只围绕一个问题：一个超级大的矩阵乘法，怎样才能算得快？从最朴素的版本一步步优化到接近 cuBLAS，顺路学完 GPU 的全部核心概念。',
    cover: 'aigc/front.png',
    published: [],
    tools: {
      'matmul-demo': '矩阵乘法演示',
      'loop-order': '循环顺序演示',
      'gpu-arch': 'NVIDIA 数据中心 GPU 架构演进',
      't4-explorer': 'T4 结构探索',
      'cuda-grid': 'CUDA 线程组织',
    },
  },
  {
    dir: 'tpu_book',
    route: 'tpu',
    title: '从零设计TPU',
    subtitle: 'Design a TPU from First Principles',
    status: '写作计划',
    blurb: '《GPU学习笔记》的另一半：假装我们是一个从零开始的芯片团队，要做一颗专门算神经网络的芯片，一步一步做设计决策，把 TPU 的每个部件都重新发明一遍。',
    cover: 'aigc/front.png',
    published: [],
  },
  {
    dir: 'ann_book',
    route: 'ann',
    title: '人人能懂的人工神经网络',
    subtitle: 'ANN for the Rest of Us',
    status: '连载中',
    blurb: '一本入门书：只用中学数学，从最简单的人工神经元入手，循序渐进搭起完整的知识体系，一直讲到 Transformer 和大语言模型。',
    cover: 'aigc/ann.png',
    // 分章文件名是 NN_slug.md，章标题写成 ##（扉页的 # 是书名），单独成页时提一级
    chapterRe: /^\d+_(.+)\.md$/,
    skipTitle: '人人能懂的人工神经网络',
    promote: true,
    published: ['02_ch00_basics.md', '03_ch01_neuron.md'],
  },
]

// 笔记文件名约定：YYYY-MM-DD-Slug.md
const NOTE_RE = /^(\d{4}-\d{2}-\d{2})-(.+)\.md$/
// 除 ann_book 外各本书的分章文件：前言、chNN_slug.md，外加后记和附录；
// _front.md / todo.md 不匹配，自然被挡在外面。前言和章节一样，列进 published 才上站。
const CHAPTER_RE = /^(_preface|ch\d+_.+|afterword|appendix|x_[a-z]_.+)\.md$/
// 分章文件的排序：前言、正文章节、后记、附录；组内按文件名
const groupOf = (name) =>
  name === '_preface.md' ? -1 : /^ch\d+_/.test(name) ? 0 : name.startsWith('afterword') ? 1 : 2

// ---- 工具函数 ----

/** 把 <img> 标签转成标准 Markdown 图片语法，并把仓库里的相对路径改成站点绝对路径。
 *  rebases 是一串 [前缀, 换成什么]，第一个匹配上的生效。
 *  转成 Markdown 语法是为了让 VitePress 自动加上 base 前缀（裸 HTML 它不管）。
 *  width 属性（像素宽度的一半，见 CLAUDE.md）借 Markdown 图片的 title 位带过去，
 *  写成 ![alt](url "width=N")，docs/.vitepress/config.mts 里的渲染钩子再把它还原成
 *  <img width="N">。Typora 写的 style="zoom:50%" 直接丢掉，图片宽度由正文栏宽兜住。 */
function normalizeImages(md, rebases) {
  const rebase = (url) => {
    for (const [from, to] of rebases) {
      if (url.startsWith(from)) return to + url.slice(from.length)
    }
    return url
  }
  return md
    .replace(/<img\s+([^>]*?)\/?>/g, (whole, attrs) => {
      const src = attrs.match(/src="([^"]+)"/)
      if (!src) return whole
      const alt = attrs.match(/alt="([^"]*)"/)
      const width = attrs.match(/\bwidth="(\d+)"/)
      const title = width ? ` "width=${width[1]}"` : ''
      return `![${alt ? alt[1] : ''}](${rebase(src[1])}${title})`
    })
    .replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (whole, alt, url) => `![${alt}](${rebase(url)})`)
}

function cleanup(md, rebases) {
  return normalizeImages(md, rebases)
    .replace(/<div style="page-break-after: always;"><\/div>/g, '') // 只对 PDF 有意义
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

/** 分章文件里章标题是 ##，单独成页时要提到 #，正文各级标题跟着上移一级。
 *  代码块里的 # 是注释，不能动。 */
function promoteHeadings(md) {
  let inFence = false
  return md
    .split('\n')
    .map((line) => {
      if (/^\s*```/.test(line)) inFence = !inFence
      return !inFence && /^#{2,6} /.test(line) ? line.slice(1) : line
    })
    .join('\n')
}

/** 收集页面内的 ## 小节，用于侧边栏展开。代码块里的 ## 不算。 */
function subsectionsOf(md) {
  const items = []
  let inFence = false
  for (const line of md.split('\n')) {
    if (/^\s*```/.test(line)) inFence = !inFence
    const h2 = !inFence && line.match(/^## (.+?)\s*$/)
    if (h2) items.push(h2[1])
  }
  return items
}

function firstHeadingOf(md) {
  return md.match(/^#{1,2} (.+?)\s*$/m)?.[1]
}

/** README 的第一段正文（H1 之后、第一个空行之前），用作工具卡片的简介。 */
function firstParagraphOf(md) {
  const lines = md.split('\n')
  let i = lines.findIndex((l) => /^# /.test(l)) + 1
  while (i < lines.length && lines[i].trim() === '') i++
  const para = []
  while (i < lines.length && lines[i].trim() !== '') para.push(lines[i++].trim())
  return para.join('')
}

// 复刻 VitePress 默认的标题锚点规则（来自 @mdit-vue/shared 的 slugify，
// 它被打包进 vitepress 里了，没法直接 import，所以照抄一份）。
// 注意 NFKD 会把全角字符归一成半角，所以「分词（Tokenization）」
// 里的（）会跟半角括号一样变成连字符，不能想当然地当成普通中文字符留着。
const rControl = /[\u0000-\u001f]/g
const rSpecial = /[\s~`!@#$%^&*()\-_+=[\]{}|\;:"'“”‘’<>,.?/]+/g
const rCombining = /[̀-ͯ]/g

function anchor(text) {
  return text
    .normalize('NFKD')
    .replace(rCombining, '')
    .replace(rControl, '')
    .replace(rSpecial, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/^(\d)/, '_$1')
    .toLowerCase()
}

function copyDir(src, dest, skip = []) {
  fs.rmSync(dest, { recursive: true, force: true })
  if (!fs.existsSync(src)) return
  fs.cpSync(src, dest, {
    recursive: true,
    filter: (f) => !f.endsWith('.DS_Store') && !skip.some((s) => f.endsWith(s)),
  })
}

const rewrites = {}

// ---- 笔记 ----

function buildNotes() {
  const srcDir = path.join(root, NOTES.dir)
  const outDir = path.join(docsDir, NOTES.dir)
  fs.rmSync(outDir, { recursive: true, force: true })
  fs.mkdirSync(outDir, { recursive: true })

  const notes = []
  for (const name of fs.readdirSync(srcDir).sort()) {
    const m = name.match(NOTE_RE)
    if (!m) continue

    const slug = name.replace(/\.md$/, '')
    const body = cleanup(fs.readFileSync(path.join(srcDir, name), 'utf8'), [
      ['./images/', `/${NOTES.route}/images/`],
    ])
    fs.writeFileSync(path.join(outDir, name), body + '\n')
    rewrites[`${NOTES.dir}/${name}`] = `${NOTES.route}/${slug}.md`

    const link = `/${NOTES.route}/${slug}`
    const subs = subsectionsOf(body)
    notes.push({
      date: m[1],
      title: firstHeadingOf(body) ?? slug,
      link,
      collapsed: subs.length > 0 ? true : undefined,
      items: subs.map((t) => ({ text: t, link: `${link}#${anchor(t)}` })),
    })
  }
  if (notes.length === 0) throw new Error(`${NOTES.dir}/ 下没有找到符合 YYYY-MM-DD-Slug.md 的文件`)

  const pub = path.join(publicDir, NOTES.route)
  copyDir(path.join(srcDir, 'images'), path.join(pub, 'images'))
  copyDir(path.join(srcDir, 'aigc'), path.join(pub, 'aigc'))
  fs.copyFileSync(path.join(srcDir, NOTES.cover), path.join(pub, 'cover.png'))

  return {
    ...NOTES,
    link: `/${NOTES.route}`,
    cover: `/${NOTES.route}/cover.png`,
    count: notes.length,
    // 侧边栏按时间正序（跟 README 一致，系列文章前后有依赖），列表页倒序
    sidebar: notes.map(({ date, title, link, collapsed, items }) => ({ text: title, link, collapsed, items })),
    latest: [...notes].reverse().map(({ date, title, link }) => ({ date, title, link })),
  }
}

// ---- 书 ----

/** 扉页 _front.md 里书名下面的第一行 > 引用，就是这本书的英文书名。 */
function subtitleFromFront(book) {
  const front = path.join(root, book.dir, 'chapters', '_front.md')
  if (!fs.existsSync(front)) throw new Error(`${book.dir} 没写 subtitle，也没有 chapters/_front.md`)
  const m = fs.readFileSync(front, 'utf8').match(/^> (?!版本)(.+)$/m)
  if (!m) throw new Error(`${book.dir}/chapters/_front.md 里没有 > 英文书名那一行`)
  return m[1].trim()
}

function buildBook(book) {
  const bookDir = path.join(root, book.dir)
  book = { ...book, subtitle: book.subtitle ?? subtitleFromFront(book) }
  const chaptersDir = path.join(bookDir, 'chapters')
  const outDir = path.join(docsDir, book.dir)
  fs.rmSync(outDir, { recursive: true, force: true })
  fs.mkdirSync(path.join(outDir, 'chapters'), { recursive: true })

  const chapterRe = book.chapterRe ?? CHAPTER_RE
  const published = new Set(book.published)
  const names = fs
    .readdirSync(chaptersDir)
    .filter((n) => chapterRe.test(n))
    .sort((a, b) => groupOf(a) - groupOf(b) || a.localeCompare(b))

  const chapters = []
  for (const name of names) {
    const raw = fs.readFileSync(path.join(chaptersDir, name), 'utf8')
    const title = firstHeadingOf(raw)
    if (!title || title === book.skipTitle) continue

    if (!published.has(name)) {
      chapters.push({ title })
      continue
    }

    const slug = name.replace(chapterRe, '$1').replace(/^_/, '').replace(/_/g, '-')
    const cleaned = cleanup(raw, [
      ['../images/', `/${book.route}/images/`],
      ['../aigc/', `/${book.route}/aigc/`],
    ])
    const body = book.promote ? promoteHeadings(cleaned) : cleaned
    fs.writeFileSync(path.join(outDir, 'chapters', name), body + '\n')
    rewrites[`${book.dir}/chapters/${name}`] = `${book.route}/${slug}.md`

    const link = `/${book.route}/${slug}`
    const subs = subsectionsOf(body)
    chapters.push({
      title,
      link,
      collapsed: subs.length > 0 ? true : undefined,
      items: subs.map((t) => ({ text: t, link: `${link}#${anchor(t)}` })),
    })
  }
  if (chapters.length === 0) throw new Error(`${book.dir}/chapters/ 下没有找到分章文件`)
  for (const name of published) {
    if (!names.includes(name)) throw new Error(`${book.dir}/chapters/${name} 不存在，但列在 published 里`)
  }

  // 插图和 AI 生成的章首图各占站点上这本书自己的子目录；封面（aigc/ 下）另拷一份成 cover.png
  const pub = path.join(publicDir, book.route)
  copyDir(path.join(bookDir, 'images'), path.join(pub, 'images'))
  copyDir(path.join(bookDir, 'aigc'), path.join(pub, 'aigc'))
  if (book.cover) fs.copyFileSync(path.join(bookDir, book.cover), path.join(pub, 'cover.png'))

  // 工具：tools/<目录>/ 下的纯静态单页，整个目录原样拷到 docs/public/<route>/tools/<目录>/，
  // 站点地址就是 /<route>/tools/<目录>/。README.md 是写给仓库读者的，不上站，
  // 只借它的第一段当卡片简介。加一个工具 = 放一个目录进 tools/，再往上面的 tools 表里加一行。
  const tools = []
  for (const [dir, title] of Object.entries(book.tools ?? {})) {
    const src = path.join(bookDir, 'tools', dir)
    if (!fs.existsSync(path.join(src, 'index.html'))) throw new Error(`${book.dir}/tools/${dir}/ 下没有 index.html`)
    copyDir(src, path.join(pub, 'tools', dir), ['README.md'])
    const readme = path.join(src, 'README.md')
    const blurb = fs.existsSync(readme) ? firstParagraphOf(fs.readFileSync(readme, 'utf8')) : ''
    tools.push({ title, blurb, link: `/${book.route}/tools/${dir}/` })
  }
  if (tools.length > 0) {
    // 工具列表页是生成的，正文是通用的一段话，卡片从 sidebar.json 里读
    fs.writeFileSync(path.join(outDir, 'tools.md'), toolsPage(book) + '\n')
    rewrites[`${book.dir}/tools.md`] = `${book.route}/tools.md`
  }

  return {
    ...book,
    chapterRe: undefined,
    tools,
    link: `/${book.route}`,
    cover: book.cover ? `/${book.route}/cover.png` : undefined,
    // 目录页要列全部章节，侧边栏只放已经上站的
    toc: chapters.map(({ title, link }) => ({ text: title, link })),
    sidebar: chapters
      .filter((c) => c.link)
      .map(({ title, link, collapsed, items }) => ({ text: title, link, collapsed, items })),
  }
}

function toolsPage(book) {
  return `# 小工具

写《${book.title}》的过程中顺手做的几个纯前端小工具，都是单个 HTML 文件，打开就能用。所有计算都在浏览器里完成，不需要后端，也不会把你的数据传到我这里来。

<ToolCards book="${book.route}" />

::: tip 源码在仓库里
每个工具的源码都在 [${book.dir}/tools/](https://github.com/zxh0/vibe-writing/tree/main/${book.dir}/tools) 下，一个目录一个工具，各自带一份 README 说明原理。把 \`index.html\` 下载到本地双击打开，效果和在线版完全一样。
:::`
}

// ---- 输出 ----

fs.mkdirSync(publicDir, { recursive: true })
const notes = buildNotes()
const books = Object.fromEntries(BOOKS.map((b) => [b.route, buildBook(b)]))

fs.mkdirSync(path.join(docsDir, '.vitepress'), { recursive: true })
fs.writeFileSync(
  path.join(docsDir, '.vitepress', 'sidebar.json'),
  JSON.stringify({ notes, books, rewrites }, null, 2) + '\n'
)

console.log(
  `生成 ${notes.count} 篇笔记，` +
    BOOKS.map((b) => {
      const k = books[b.route]
      const t = k.tools.length ? `+${k.tools.length} 个工具` : ''
      return `${b.route}: ${k.sidebar.length}/${k.toc.length} 章${t}`
    }).join('，')
)
