import { localizePath } from "@/i18n/routes";
import type { SiteLocale } from "@/i18n/config";
import { getBlogPosts } from "@/utils/blogCollection";

// 获取文章分类
const getCategories = async (locale: SiteLocale) => {
  const posts = (await getBlogPosts(locale)).filter((i) => !i.data.hide);
  const categoriesList = posts.reduce<Record<string, number>>((acc, i) => {
    acc[i.data.categories] = (acc[i.data.categories] || 0) + 1;
    return acc;
  }, {});
  return Object.entries(categoriesList).map(([title, count]) => ({ title, count }));
}

// 获取统计数据
const getCountInfo = async (locale: SiteLocale) => {
  const posts = (await getBlogPosts(locale)).filter((i) => !i.data.hide);
  const categories = await getCategories(locale);
  const tags = await getTags(locale);
  return { ArticleCount: posts.length, CategoryCount: categories.length, TagCount: tags.length }
}

// 获取文章标签
const getTags = async (locale: SiteLocale) => {
  const posts = (await getBlogPosts(locale)).filter((i) => !i.data.hide);
  const tagList = posts.reduce<Record<string, number>>((acc, i) => {
    (i.data.tags || []).forEach((tag: string) => {
      acc[tag] = (acc[tag] || 0) + 1;
    });
    return acc;
  }, {});
  return Object.entries(tagList).sort((a, b) => b[1] - a[1]);
}

// 获取推荐文章 (给文章添加 recommend: true 字段)
const getRecommendArticles = async (locale: SiteLocale) => {
  const posts = (await getBlogPosts(locale)).filter(i => !i.data.hide);
  const recommendList = posts.filter(i => i.data.recommend);
  return (recommendList.length ? recommendList : posts.slice(0, 6)).map(i => ({ title: i.data.title, date: i.data.date, id: i.data.id, href: localizePath(`/article/${i.data.id}`, locale) }))
};

// 获取上一篇下一篇文章
const getPrevNextPosts = async (locale: SiteLocale, id: string, emptyTitle: string) => {
  const posts = await getBlogPosts(locale);
  const noHidePosts = posts.filter(i => !i.data.hide);
  noHidePosts.sort((a, b) => a.data.date.valueOf() - b.data.date.valueOf());
  const index = noHidePosts.findIndex(i => String(i.data.id) === id);
  const mapPost = (post?: typeof noHidePosts[number]) => post ? { ...post.data, href: localizePath(`/article/${post.data.id}`, locale) } : { title: emptyTitle, href: "#" };
  return { prev: mapPost(noHidePosts[index - 1]), next: mapPost(noHidePosts[index + 1]) }
}


export { getCategories, getTags, getRecommendArticles, getCountInfo, getPrevNextPosts };
