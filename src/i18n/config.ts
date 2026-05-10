export const DEFAULT_LOCALE = "zh-CN";
export const LOCALES = [DEFAULT_LOCALE, "en"] as const;

export type SiteLocale = (typeof LOCALES)[number];
export type BlogCollectionName = "blog" | "blogEn";

export const LOCALE_SEGMENT: Record<SiteLocale, string> = {
  "zh-CN": "",
  en: "en",
};

export const BLOG_COLLECTION_BY_LOCALE: Record<SiteLocale, BlogCollectionName> = {
  "zh-CN": "blog",
  en: "blogEn",
};
