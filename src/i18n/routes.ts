import { BLOG_COLLECTION_BY_LOCALE, DEFAULT_LOCALE, LOCALES, LOCALE_SEGMENT, type BlogCollectionName, type SiteLocale } from "@/i18n/config";

const trimTrailingSlash = (pathname: string) => pathname !== "/" ? pathname.replace(/\/+$/, "") : pathname;

export const getLocaleFromPathname = (pathname: string): SiteLocale => {
  const normalizedPath = trimTrailingSlash(pathname);
  return normalizedPath === "/en" || normalizedPath.startsWith("/en/") ? "en" : DEFAULT_LOCALE;
};

export const getBlogCollectionName = (locale: SiteLocale): BlogCollectionName => BLOG_COLLECTION_BY_LOCALE[locale];

export const getLocaleSegment = (locale: SiteLocale) => LOCALE_SEGMENT[locale];

export const stripLocalePrefix = (pathname: string) => {
  const normalizedPath = trimTrailingSlash(pathname);
  if (normalizedPath === "/en") return "/";
  if (normalizedPath.startsWith("/en/")) return normalizedPath.slice(3) || "/";
  return normalizedPath || "/";
};

export const localizePath = (pathname: string, locale: SiteLocale) => {
  const cleanPath = pathname.startsWith("/") ? pathname : `/${pathname}`;
  const normalizedPath = cleanPath === "/" ? "/" : trimTrailingSlash(cleanPath);
  const segment = getLocaleSegment(locale);
  return segment ? (normalizedPath === "/" ? `/${segment}` : `/${segment}${normalizedPath}`) : normalizedPath;
};

export const switchLocalePath = (pathname: string, locale: SiteLocale) => localizePath(stripLocalePrefix(pathname), locale);

export const isDefaultLocale = (locale: SiteLocale) => locale === DEFAULT_LOCALE;

export const getOtherLocales = (locale: SiteLocale) => LOCALES.filter((item) => item !== locale);
