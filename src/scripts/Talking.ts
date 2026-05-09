
import vh from 'vh-plugin'
import { fmtDate } from '@/utils/index'
import { $GET } from '@/utils/index'
import type { TalkingItem, ToolSource } from '@/type/tool'
// 图片懒加载
import vhLzImgInit from "@/scripts/vhLazyImg";
import Site_INFO from '@/config'

const TalkingInit = async (data: ToolSource<TalkingItem>) => {
  const talkingDOM = document.querySelector('.main-inner-content>.vh-tools-main>main.talking-main')
  if (!talkingDOM) return;
  try {
    let res = data;
    if (typeof data === 'string') {
      const remoteData = await $GET<TalkingItem[]>(data);
      if (!remoteData) throw new Error('talking data is empty');
      res = remoteData;
    }
    talkingDOM.innerHTML = res.map((i) => `<article><header><img data-vh-lz-src="/assets/images/avatar.jpg" /><p class="info"><span>${Site_INFO.Author}</span><time>${fmtDate(i.date)}前</time></p></header><section class="main">${i.content}</section><footer>${i.tags.map((tag) => `<span>${tag}</span>`).join('')}</footer></article>`).join('');
    // 图片懒加载
    vhLzImgInit();
  } catch {
    vh.Toast('获取数据失败')
  }
}


// 动态说说初始化
import TALKING_DATA from "@/page_data/Talking";
const { api, data } = TALKING_DATA;
export default () => TalkingInit(api || data);
