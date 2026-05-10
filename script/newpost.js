import path from 'path';
import dayjs from 'dayjs';
import crypto from 'crypto';
import { fileURLToPath } from 'url';
import { promises as fs } from 'fs';
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// 获取命令行参数
const articleName = process.argv.slice(2).join(' ').trim();
const articleID = crypto.createHash('sha256').update(dayjs().valueOf().toString()).digest('hex').slice(0, 16);
if (!articleName) {
  console.error('请提供文章名称，例如：pnpm newpost "第一篇文章"');
  process.exit(1);
}
const banners = await fs.readdir("./public/assets/images/banner")
const banner = banners[Math.floor(Math.random()*banners.length)]
const articleDate = dayjs().format('YYYY-MM-DD HH:mm:ss');

const zhArticleContent = `---
title: "${articleName.replace(/"/g, '\\"')}"
categories: "分类"
tags: ["标签"]
id: "${articleID}"
date: ${articleDate}
cover: "/assets/images/banner/${banner}"
---

:::note
文章描述
:::

### 标题1

::btn[按钮]{link="链接" type="info"}`;

const enPlaceholderContent = `---
title: "${articleName.replace(/"/g, '\\"')}"
categories: "category"
tags: ["tag"]
id: "${articleID}"
date: ${articleDate}
cover: "/assets/images/banner/${banner}"
hide: true
---

:::note{type="info"}
TODO: add English version.
:::
`;

const ensureNotExists = async (filePath) => {
  try {
    await fs.access(filePath);
    console.error(`❌ 文件已存在：${filePath}`);
    process.exit(1);
  } catch (error) {
    if (error.code !== 'ENOENT') {
      throw error;
    }
  }
};

const init = async () => {
  const now = dayjs();
  const zhFilePath = path.join(__dirname, '../src/content/blog', `${articleName}.md`);
  const enFilePath = path.join(__dirname, '../src/content/blog-en', `${articleName}.md`);
  try {
    await ensureNotExists(zhFilePath);
    await ensureNotExists(enFilePath);
    await fs.mkdir(path.dirname(zhFilePath), { recursive: true });
    await fs.mkdir(path.dirname(enFilePath), { recursive: true });
    await fs.writeFile(zhFilePath, zhArticleContent, 'utf8');
    await fs.writeFile(enFilePath, enPlaceholderContent, 'utf8');
    console.log('✅ 文章创建成功');
    console.log(`📅 日期：${now.format('YYYY-MM-DD')}`);
    console.log(`📂 中文路径：${zhFilePath}`);
    console.log(`📂 英文占位路径：${enFilePath}`);
    console.log(`🆔 ID：${articleID.slice(0, 16)} (可手动修改)`);
  } catch (error) {
    console.error('❌ 创建失败：');
    console.error(`错误类型：${error.code || 'UNKNOWN_ERROR'}`);
    console.error(`详细信息：${error.message}`);
    process.exit(1);
  }
}
init();
