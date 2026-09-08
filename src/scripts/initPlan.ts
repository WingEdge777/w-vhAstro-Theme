export type InitFeature =
  | "site-time"
  | "back-top"
  | "smooth-scroll"
  | "view-image"
  | "code"
  | "lazy-image"
  | "live-photo"
  | "video"
  | "music"
  | "links"
  | "talking"
  | "google-ad"
  | "seo-push"
  | "umami"
  | "comment"
  | "type-write"
  | "paopao"
  | "search-preload"
  | "search-ui"
  | "mobile-sidebar";

const ROUTE_FEATURES: InitFeature[] = [
  "code",
  "lazy-image",
  "live-photo",
  "video",
  "music",
  "links",
  "talking",
  "google-ad",
  "seo-push",
];

export const getInitFeatures = (firstLoad: boolean, hasComment: boolean): InitFeature[] => [
  ...(firstLoad ? ["site-time", "back-top", "smooth-scroll", "view-image"] satisfies InitFeature[] : []),
  ...ROUTE_FEATURES,
  // 首屏由 Umami script 自动统计；仅在 Swup 切页时手动上报
  ...(!firstLoad ? ["umami"] satisfies InitFeature[] : []),
  ...(hasComment ? ["comment"] satisfies InitFeature[] : []),
  ...(firstLoad ? ["type-write"] satisfies InitFeature[] : []),
  "paopao",
  ...(firstLoad ? ["search-preload"] satisfies InitFeature[] : []),
  "search-ui",
  "mobile-sidebar",
];
