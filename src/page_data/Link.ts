import type { LinkItem, ToolDataSource } from "@/type/tool";

export default {
  // API 接口请求优先，数据格式保持和 data 一致
  api: '',
  // api 为空则使用 data 静态数据
  data: [
    {
      "name": "WingEdge777的博客",
      "link": "https://www.wingedge777.com",
      "avatar": "/assets/images/avatar.jpg",
      "descr": "该想点什么东西呢"
    },
    {
      "name": "韩小韩博客",
      "link": "https://www.vvhan.com/",
      "avatar": "https://q1.qlogo.cn/g?b=qq&nk=1655466387&s=640",
      "descr": "运气是计划之外的东西."
    }
  ]
} satisfies ToolDataSource<LinkItem>;
