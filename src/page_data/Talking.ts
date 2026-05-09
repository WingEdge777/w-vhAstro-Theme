import type { TalkingItem, ToolDataSource } from "@/type/tool";

export default {
  // API 接口请求优先，数据格式保持和 data 一致
  api: '',
  // api 为空则使用 data 静态数据 
  // 注意：图片请用 vh-img-flex 类包裹
  data: [
    {
      "date": "2025-11-06 15:36:16",
      "tags": [
        "code",
      ],
      "content": "打工才是生活的常态"
    }
  ]
} satisfies ToolDataSource<TalkingItem>;
