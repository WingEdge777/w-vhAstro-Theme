const LANG_RESOLVED_KEY = "lang_resolved";
const GEOIP_API = "https://ipapi.co/json/";
const TIMEOUT_MS = 3000;

const isOnEnPath = () =>
  location.pathname === "/en" || location.pathname.startsWith("/en/");

const detectAndRedirect = () => {
  if (sessionStorage.getItem(LANG_RESOLVED_KEY)) return;

  const xhr = new XMLHttpRequest();
  xhr.open("GET", GEOIP_API, true);
  xhr.timeout = TIMEOUT_MS;

  xhr.onload = () => {
    if (xhr.status !== 200) {
      sessionStorage.setItem(LANG_RESOLVED_KEY, "1");
      return;
    }
    try {
      const data = JSON.parse(xhr.responseText);
      const isCN = data && data.country_code === "CN";
      const onEN = isOnEnPath();
      if (isCN && onEN) {
        location.replace(location.origin + "/");
      } else if (!isCN && !onEN) {
        location.replace(location.origin + "/en");
      }
      sessionStorage.setItem(LANG_RESOLVED_KEY, "1");
    } catch {
      sessionStorage.setItem(LANG_RESOLVED_KEY, "1");
    }
  };

  xhr.onerror = () => sessionStorage.setItem(LANG_RESOLVED_KEY, "1");
  xhr.ontimeout = () => sessionStorage.setItem(LANG_RESOLVED_KEY, "1");
  xhr.send();
};

detectAndRedirect();
