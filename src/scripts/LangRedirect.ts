const LANG_RESOLVED_KEY = "lang_resolved";
const GEOIP_API = "https://ipapi.co/json/";
const TIMEOUT_MS = 3000;
const TTL_MS = 24 * 60 * 60 * 1000;

const isOnEnPath = () =>
  location.pathname === "/en" || location.pathname.startsWith("/en/");

const switchPathLocale = (toEN: boolean) => {
  const path = location.pathname;
  if (toEN) {
    return "/en" + (path === "/" ? "" : path);
  }
  if (path === "/en") return "/";
  if (path.startsWith("/en/")) return path.slice(3) || "/";
  return path;
};

const isResolved = () => {
  const ts = localStorage.getItem(LANG_RESOLVED_KEY);
  if (!ts) return false;
  return Date.now() - Number(ts) < TTL_MS;
};

const resolve = () => localStorage.setItem(LANG_RESOLVED_KEY, String(Date.now()));

const isBot = () => /bot|crawl|spider|slurp|mediapartners|preview/i.test(navigator.userAgent);

const detectAndRedirect = () => {
  if (isBot() || isResolved()) return;

  const xhr = new XMLHttpRequest();
  xhr.open("GET", GEOIP_API, true);
  xhr.timeout = TIMEOUT_MS;

  xhr.onload = () => {
    if (xhr.status !== 200) { resolve(); return; }
    try {
      const data = JSON.parse(xhr.responseText);
      const isCN = data && data.country_code === "CN";
      const onEN = isOnEnPath();
      if (isCN && onEN) {
        resolve();
        location.replace(location.origin + switchPathLocale(false));
      } else if (!isCN && !onEN) {
        resolve();
        location.replace(location.origin + switchPathLocale(true));
      } else {
        resolve();
      }
    } catch {
      resolve();
    }
  };

  xhr.onerror = resolve;
  xhr.ontimeout = resolve;
  xhr.send();
};

detectAndRedirect();
