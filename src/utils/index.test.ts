import { describe, expect, it } from "vitest";
import { fmtPage, getDescription } from "./index";

describe("utils/index", () => {
  it("优先使用 frontmatter description", () => {
    const text = getDescription({
      body: "正文内容",
      data: { description: "自定义 SEO 摘要" },
    });

    expect(text).toBe("自定义 SEO 摘要");
  });

  it("frontmatter description 为空时回退到自动截取", () => {
    const text = getDescription({
      body: "Hello World",
      data: { description: "   " },
    });

    expect(text).toBe("Hello World");
  });

  it("优先从 rendered html 提取摘要", () => {
    const text = getDescription({
      body: "",
      rendered: { html: "<p>Hello <strong>World</strong></p>" },
    });

    expect(text).toBe("HelloWorld");
  });

  it("body 为空时返回默认摘要", () => {
    const text = getDescription({
      body: "",
    });

    expect(text).toBe("暂无简介");
  });

  it("格式化页码", () => {
    expect(fmtPage("/2/")).toBe("2");
    expect(fmtPage(undefined)).toBeNull();
  });
});
