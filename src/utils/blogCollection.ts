import { getCollection } from "astro:content";
import { getBlogCollectionName } from "@/i18n/routes";
import type { BlogEntry } from "@/type/blog";
import type { SiteLocale } from "@/i18n/config";

export const getBlogPosts = async (locale: SiteLocale): Promise<BlogEntry[]> => {
  const collection = getBlogCollectionName(locale);
  const posts = await getCollection(collection);
  return posts.sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
};
