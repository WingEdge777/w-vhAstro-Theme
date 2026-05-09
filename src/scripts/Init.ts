import { inRouter, outRouter } from "@/utils/updateRouter";
import { getInitFeatures, type InitFeature } from "@/scripts/initPlan";
import type { Destroyable } from "@/type/common";
// Banner 打字效果
import TypeWriteInit from "@/scripts/TypeWrite";
// 泡泡🫧效果
import PaoPaoInit from "@/scripts/PaoPao";
// 初始化文章代码块
import codeInit from "@/scripts/Code";
// 初始化视频播放器
import videoInit from "@/scripts/Video";
// 初始化音乐播放器
import musicInit from "@/scripts/Music";
// 初始化 LivePhoto
import livePhotoInit from '@/scripts/LivePhoto'
// 初始化BackTop组件
import BackTopInitFn from "@/scripts/BackTop";
// 搜索
import { searchFn, vhSearchInit } from "@/scripts/Search";
// 图片懒加载
import vhLzImgInit from "@/scripts/vhLazyImg";
// 图片灯箱
import ViewImage from "@/scripts/ViewImage";
// 底部网站运行时间
import initWebSiteTime from "@/scripts/Footer";
// 友情链接初始化
import initLinks from "@/scripts/Links";
// 动态说说初始化
import initTalking from "@/scripts/Talking";
// 文章评论初始化
import { checkComment, commentInit } from "@/scripts/Comment";
// 移动端侧边栏初始化
import initMobileSidebar from "@/scripts/MobileSidebar";
// Google 广告
import GoogleAdInit from "@/scripts/GoogleAd";
// Han Analytics 统计
//  谷歌 SEO 推送
import SeoPushInit from "@/scripts/SeoPush";
// SmoothScroll 滚动优化
import SmoothScroll from "@/scripts/Smoothscroll";

// ============================================================

const videoList: Destroyable[] = [];
const musicList: Destroyable[] = [];

const featureHandlers: Record<InitFeature, () => void | Promise<void>> = {
  "site-time": initWebSiteTime,
  "back-top": BackTopInitFn,
  "smooth-scroll": SmoothScroll,
  "view-image": ViewImage,
  "code": codeInit,
  "lazy-image": vhLzImgInit,
  "live-photo": livePhotoInit,
  "video": () => videoInit(videoList),
  "music": () => musicInit(musicList),
  "links": initLinks,
  "talking": initTalking,
  "google-ad": GoogleAdInit,
  "seo-push": SeoPushInit,
  "comment": () => {
    const commentKey = checkComment();
    if (commentKey) return commentInit(commentKey);
  },
  "type-write": TypeWriteInit,
  "paopao": PaoPaoInit,
  "search-preload": () => searchFn(""),
  "search-ui": vhSearchInit,
  "mobile-sidebar": initMobileSidebar,
};

const runInitFeatures = async (firstLoad: boolean) => {
  const features = getInitFeatures(firstLoad, Boolean(checkComment()));
  for (const feature of features) {
    await featureHandlers[feature]();
  }
};

const disposeMediaPlayers = () => {
  videoList.forEach((item) => item.destroy());
  videoList.length = 0;
  musicList.forEach((item) => item.destroy());
  musicList.length = 0;
};

export default () => {
  // 首次初始化
  runInitFeatures(true);
  // 进入页面时触发
  inRouter(() => runInitFeatures(false));
  // 离开当前页面时触发
  outRouter(disposeMediaPlayers);
}
