import fs from 'fs/promises';
import path from 'path';
import * as cheerio from 'cheerio';
import { DEFAULT_LOCALE, type SiteLocale } from '@/i18n/config';
import { localizePath } from '@/i18n/routes';
import type { BlogEntry } from '@/type/blog';

const getSearchFileName = (locale: SiteLocale) => locale === DEFAULT_LOCALE ? 'vh-search.json' : `vh-search.${locale}.json`;

export default async (posts: BlogEntry[], locale: SiteLocale) => {
  const searchIndex = posts.filter((i) => !i.data.hide).map(i => {
    const $ = cheerio.load(`<body>${i.rendered.html}</body>`);
    return {
      title: i.data.title,
      url: localizePath(`/article/${i.data.id}`, locale),
      content: `${i.data.title} - ` + $('body').text().replace(/\n/g, '').replace(/<[^>]+>/g, '')
    };
  });

  const fileName = getSearchFileName(locale);
  try {
    await fs.writeFile(
      path.join(process.cwd(), 'dist', fileName),
      JSON.stringify(searchIndex)
    );
    await fs.writeFile(
      path.join(process.cwd(), 'public', fileName),
      JSON.stringify(searchIndex)
    );
    console.log('\x1b[32m%s\x1b[0m', `搜索文件 ${fileName} 已生成 successfully`);
  } catch (error) {
    console.error('Error writing search index file:', error);
  }
};
