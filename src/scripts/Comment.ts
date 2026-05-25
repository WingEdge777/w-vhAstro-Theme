
import SITE_INFO from "@/config";
import { LoadScript } from "@/utils/index";
import fetchPV from "@/scripts/ArticlePV";
declare const twikoo: any;

const TwikooFn = async (commentDOM: string) => {
  document.querySelector(commentDOM)!.innerHTML = '<section class="vh-space-loading"><span></span><span></span><span></span></section>'
  await LoadScript("https://cdn.jsdelivr.net/npm/twikoo@1.7.9/dist/twikoo.min.js");
  twikoo.init({ envId: SITE_INFO.Comment.Twikoo.envId, el: commentDOM, onCommentLoaded: () => { setTimeout(() => document.querySelectorAll('.vh-comment a[href="#"]').forEach(link => link.removeAttribute('href'))); fetchPV(); } })
}

// 检查是否开启评论
const checkComment = () => {
  const CommentARR: any = Object.keys(SITE_INFO.Comment);
  const CommentItem = CommentARR.find((i: keyof typeof SITE_INFO.Comment) => SITE_INFO.Comment[i].enable);
  return CommentItem;
}

// 初始化评论插件
const commentInit = async (key: string) => {
  // 评论 DOM 
  const commentDOM = '.vh-comment>section'
  if (!document.querySelector(commentDOM)) return;
  // 评论列表
  const CommentList: any = { TwikooFn, };
  // 初始化评论
  CommentList[`${key}Fn`](commentDOM);
}

export { checkComment, commentInit }