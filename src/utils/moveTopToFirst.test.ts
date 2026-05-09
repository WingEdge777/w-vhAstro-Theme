import { describe, expect, it } from "vitest";
import moveTopToFirst from "./moveTopToFirst";

describe("moveTopToFirst", () => {
  it("把置顶文章移到最前", () => {
    const posts = [
      { data: { id: 1, top: false } },
      { data: { id: 2, top: true } },
      { data: { id: 3, top: false } },
    ];

    const result = moveTopToFirst(posts);

    expect(result[0]?.data.id).toBe(2);
    expect(result).toHaveLength(3);
  });

  it("没有置顶时保持原顺序", () => {
    const posts = [
      { data: { id: 1 } },
      { data: { id: 2 } },
    ];

    expect(moveTopToFirst(posts).map((item) => item.data.id)).toEqual([1, 2]);
  });
});
