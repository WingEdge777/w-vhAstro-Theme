import SITE_INFO from "@/config";

const PV_ELEMENT_ID = "twikoo_visitors";

const fetchPV = async () => {
  const el = document.getElementById(PV_ELEMENT_ID);
  if (!el) return;

  const proxyApi = SITE_INFO.Comment.Twikoo?.proxyPath;
  if (!proxyApi) return;

  try {
    const res = await fetch(proxyApi, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        event: "COUNTER_GET",
        url: window.location.pathname,
        href: window.location.href,
        title: document.title,
      }),
    });
    if (!res.ok) return;
    const data = await res.json();
    const count = data?.result?.time;
    if (count !== undefined && count !== null) {
      el.textContent = String(count);
    }
  } catch {
    // silently fail
  }
};

export default fetchPV;
