const PV_ELEMENT_ID = "twikoo_visitors";

const fetchPV = async (envId: string) => {
  const el = document.getElementById(PV_ELEMENT_ID);
  if (!el) return;

  try {
    const res = await fetch(envId, {
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
