import SITE_INFO from "@/config";
declare const twikoo: any;

console.log("[ArticlePV] module loaded");

const PV_ELEMENT_ID = "vh-pv";

const fetchPV = async () => {
  console.log("[ArticlePV] fetchPV called");
  const el = document.getElementById(PV_ELEMENT_ID);
  if (!el) { console.log("[ArticlePV] element #vh-pv not found"); return; }

  const envId = SITE_INFO.Comment.Twikoo?.envId;
  if (!envId) { console.log("[ArticlePV] envId is empty"); return; }

  try {
    console.log("[ArticlePV] calling twikoo.getVisitorCount, urls:", [location.pathname], "envId:", envId);
    const result = await twikoo.getVisitorCount({
      urls: [location.pathname],
      envId,
    });
    console.log("[ArticlePV] getVisitorCount result:", JSON.stringify(result));
    const count = result?.[0]?.visitorCount ?? 0;
    console.log("[ArticlePV] parsed count:", count);
    const span = el.querySelector("span");
    if (span) span.textContent = `${count} views`;
  } catch (e) {
    console.error("[ArticlePV] getVisitorCount error:", e);
    const span = el.querySelector("span");
    if (span) span.textContent = "-- views";
  }
};

export default fetchPV;
