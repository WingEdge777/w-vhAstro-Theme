import { getCollection } from "astro:content";
import { DEFAULT_LOCALE, LOCALES, type SiteLocale } from "@/i18n/config";
import { getBlogCollectionName } from "@/i18n/routes";

const staticPages = ["/", "/about", "/links", "/message", "/archives", "/talking"];

const buildLocaleRoutes = async (locale: SiteLocale): Promise<Set<string>> => {
  const routes = new Set(staticPages);
  const collection = getBlogCollectionName(locale);
  const posts = await getCollection(collection);
  for (const post of posts) {
    if (!post.data.hide) {
      routes.add(`/article/${post.data.id}`);
      routes.add(`/categories/${post.data.categories}`);
      for (const tag of post.data.tags || []) {
        routes.add(`/tag/${tag}`);
      }
    }
  }
  return routes;
};

let _cache: Record<SiteLocale, Set<string>> | null = null;

const getLocaleRoutes = async (): Promise<Record<SiteLocale, Set<string>>> => {
  if (!_cache) {
    _cache = {
      [DEFAULT_LOCALE]: await buildLocaleRoutes(DEFAULT_LOCALE),
      en: await buildLocaleRoutes("en"),
    };
  }
  return _cache;
};

export const getAvailableLocales = async (pathname: string): Promise<SiteLocale[]> => {
  const routes = await getLocaleRoutes();
  const cleanPath = pathname === "/" ? "/" : pathname.replace(/\/+$/, "");
  const stripped = cleanPath.startsWith("/en") ? cleanPath.slice(3) || "/" : cleanPath;
  return LOCALES.filter((locale) => routes[locale].has(stripped));
};
