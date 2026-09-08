import SITE_INFO from "@/config";

declare global {
  interface Window {
    umami?: {
      track: (payload?: Record<string, unknown> | ((props: Record<string, unknown>) => Record<string, unknown>)) => void;
    };
  }
}

// Swup 前端切页不会触发 Umami 默认 pageview，需手动上报
export default () => {
  if (!SITE_INFO.Umami?.enable) return;
  window.umami?.track((props) => ({
    ...props,
    url: window.location.pathname + window.location.search,
    title: document.title,
  }));
};
