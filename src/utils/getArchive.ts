
import type { SiteLocale } from "@/i18n/config";
import type { ArchiveGroup, BlogEntry } from "@/type/blog";
import { getBlogPosts } from "@/utils/blogCollection";

// 格式化文章列表
const fmtArticleList = (articleList: BlogEntry[]): ArchiveGroup[] => {
  // 按年份分类
  const groupedByYear = articleList.reduce<Record<number, BlogEntry["data"][]>>((acc, item) => {
    const year = item.data.date.getFullYear();
    // 初始化
    !acc[year] && (acc[year] = []);
    acc[year].push(item.data);
    return acc;
  }, {});
  // 转换为目标格式
  return Object.keys(groupedByYear).map(year => ({ name: parseInt(year), data: groupedByYear[year] })).reverse();
}

// 获取分类下的文章列表
const getCategoriesList = async (locale: SiteLocale, categories: string) => {
  const posts = await getBlogPosts(locale);
  const articleList = posts.filter((i) => !i.data.hide && i.data.categories == categories).sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
  return fmtArticleList(articleList);
}

// 获取标签下的文章列表
const getTagsList = async (locale: SiteLocale, tags: string) => {
  const posts = await getBlogPosts(locale);
  const articleList = posts.filter((i) => !i.data.hide && (i.data.tags || []).map((_i) => (String(_i))).includes(tags)).sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
  return fmtArticleList(articleList);
}

// 获取归档列表
const getArchiveList = async (locale: SiteLocale) => {
  const posts = await getBlogPosts(locale);
  const articleList = posts.filter((i) => !i.data.hide).sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
  return fmtArticleList(articleList);
}

export { getCategoriesList, getTagsList, getArchiveList };
