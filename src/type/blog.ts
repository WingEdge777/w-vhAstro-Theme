import type { CollectionEntry } from "astro:content";

export type BlogEntry = CollectionEntry<"blog">;
export type BlogData = BlogEntry["data"];

export type RenderableBlogEntry = Pick<BlogEntry, "body"> & {
  rendered?: { html: string } | null;
};

export type ArchiveGroup = {
  name: number;
  data: BlogData[];
};
