import { describe, expect, it } from "vitest";
import { getInitFeatures } from "./initPlan";

describe("getInitFeatures", () => {
  it("首屏包含 only-first 特性", () => {
    expect(getInitFeatures(true, true)).toEqual([
      "site-time",
      "back-top",
      "smooth-scroll",
      "view-image",
      "code",
      "lazy-image",
      "live-photo",
      "video",
      "music",
      "links",
      "talking",
      "google-ad",
      "seo-push",
      "comment",
      "type-write",
      "paopao",
      "search-preload",
      "search-ui",
      "mobile-sidebar",
    ]);
  });

  it("路由切换不重复首屏特性", () => {
    expect(getInitFeatures(false, false)).toEqual([
      "code",
      "lazy-image",
      "live-photo",
      "video",
      "music",
      "links",
      "talking",
      "google-ad",
      "seo-push",
      "paopao",
      "search-ui",
      "mobile-sidebar",
    ]);
  });
});
