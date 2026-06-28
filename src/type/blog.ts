import type { CollectionEntry } from "astro:content";

export type BlogCollectionKey = "blog" | "blogEn";
export type BlogEntry = CollectionEntry<"blog"> | CollectionEntry<"blogEn">;
export type BlogData = BlogEntry["data"];

export type RenderableBlogEntry = Pick<BlogEntry, "body"> & {
  data?: Pick<BlogData, "description">;
  rendered?: { html: string } | null;
};

export type ArchiveGroup = {
  name: number;
  data: BlogData[];
};
