import { DEFAULT_LOCALE, type SiteLocale } from "@/i18n/config";

type Dictionary = {
  site: {
    title: string;
    subtitle: string;
    description: string;
  };
  header: {
    home: string;
    search: string;
    localeLabel: string;
    nav: {
      blog: string;
      links: string;
      talking: string;
      archives: string;
      message: string;
      about: string;
    };
  };
  search: {
    placeholder: string;
  };
  aside: {
    articleCount: string;
    categoryCount: string;
    tagCount: string;
    categories: string;
    popularTags: string;
    recommendArticles: string;
    toc: string;
    ad: string;
  };
  archive: {
    articleCountSuffix: string;
  };
  pagination: {
    prev: string;
    next: string;
    first: string;
    page: (page: string) => string;
  };
  listPage: {
    pageTitle: (page: string) => string;
  };
  categoryPage: {
    title: (name: string) => string;
  };
  tagPage: {
    title: (name: string) => string;
  };
  archivesPage: {
    title: string;
  };
  footer: {
    uptime: string;
  };
  copyright: {
    publishedAt: (author: string, time: string) => string;
    articleLink: string;
    licensePrefix: string;
    licenseSuffix: (siteName: string) => string;
  };
  typewrite: string[];
  article: {
    noMore: string;
  };
};

const dictionaries: Record<SiteLocale, Dictionary> = {
  "zh-CN": {
    site: {
      title: "WingEdge777's Blog",
      subtitle: "z frontier",
      description: "这是 WingEdge777 的博客，专注于算法和后端相关技术的实战分享，涵盖 LLM、AIGC、AI Infra 等技术优化，涉及Python、C++、Go、Rust、Linux、Docker、k8s等领域。同时，博客也分享作者的生活、想法随笔等。",
    },
    header: {
      home: "首页",
      search: "搜索",
      localeLabel: "语言",
      nav: {
        blog: "博客",
        links: "友链",
        talking: "动态",
        archives: "归档",
        message: "留言",
        about: "关于",
      },
    },
    search: {
      placeholder: "搜索文章...",
    },
    aside: {
      articleCount: "文章数",
      categoryCount: "分类数",
      tagCount: "标签数",
      categories: "分类",
      popularTags: "热门标签",
      recommendArticles: "推荐文章",
      toc: "文章目录",
      ad: "广而告之",
    },
    archive: {
      articleCountSuffix: "篇文章",
    },
    pagination: {
      prev: "上一页",
      next: "下一页",
      first: "第一页",
      page: (page) => `第${page}页`,
    },
    listPage: {
      pageTitle: (page) => `第${page}页文章`,
    },
    categoryPage: {
      title: (name) => `分类 ${name} 下的文章`,
    },
    tagPage: {
      title: (name) => `标签 ${name} 下的文章`,
    },
    archivesPage: {
      title: "归档",
    },
    footer: {
      uptime: "稳定运行",
    },
    copyright: {
      publishedAt: (author, time) => `本文由 ${author} 于 ${time} 发布`,
      articleLink: "文章地址：",
      licensePrefix: "本博客所有文章除特别声明外，均采用 ",
      licenseSuffix: (siteName) => ` 许可协议。完整转载请注明来自 ${siteName}！`,
    },
    typewrite: ["活到老，学到老", "stay hungry, stay foolish"],
    article: {
      noMore: "没有啦~",
    },
  },
  en: {
    site: {
      title: "WingEdge777's Blog",
      subtitle: "z frontier",
      description: "WingEdge777's blog on algorithms, backend engineering, AI infra, and personal notes.",
    },
    header: {
      home: "Home",
      search: "Search",
      localeLabel: "Language",
      nav: {
        blog: "Blog",
        links: "Links",
        talking: "Moments",
        archives: "Archives",
        message: "Message",
        about: "About",
      },
    },
    search: {
      placeholder: "Search articles...",
    },
    aside: {
      articleCount: "Posts",
      categoryCount: "Categories",
      tagCount: "Tags",
      categories: "Categories",
      popularTags: "Popular Tags",
      recommendArticles: "Recommended",
      toc: "Contents",
      ad: "Sponsored",
    },
    archive: {
      articleCountSuffix: "posts",
    },
    pagination: {
      prev: "Previous",
      next: "Next",
      first: "First page",
      page: (page) => `Page ${page}`,
    },
    listPage: {
      pageTitle: (page) => `Posts - Page ${page}`,
    },
    categoryPage: {
      title: (name) => `Posts in ${name}`,
    },
    tagPage: {
      title: (name) => `Posts tagged ${name}`,
    },
    archivesPage: {
      title: "Archives",
    },
    footer: {
      uptime: "Uptime",
    },
    copyright: {
      publishedAt: (author, time) => `Published by ${author} on ${time}`,
      articleLink: "Article URL: ",
      licensePrefix: "Unless stated otherwise, all articles on this blog are licensed under ",
      licenseSuffix: (siteName) => `. Please credit ${siteName} when reposting.`,
    },
    typewrite: ["Keep learning.", "Stay hungry, stay foolish."],
    article: {
      noMore: "No more posts",
    },
  },
};

export const getDictionary = (locale: SiteLocale = DEFAULT_LOCALE) => dictionaries[locale];
