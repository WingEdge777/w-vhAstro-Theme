---
title: "leetcode 3321. 计算子数组的 x-sum II (hard)"
categories: "leetcode"
tags: 
    - "双指针"
    - "滑动窗口"
id: "b084669b799d0d42"
date: 2025-11-15 12:30:41
cover: "/assets/images/banner/8cef6fb3c78dc3ad.webp"
---

:::note
是 leetcode 每日一题跳出来的。咋看了一眼题就觉得很有趣，让我想到经典的类似题，如：维护一个队列中的中位数，具体题不记得了。
:::

## 计算子数组的 x-sum II

如之前所言，这题的题面就让人容易联想到维护滑动窗口中的中位数，维护中位数的做法是如何实现的呢，就是用一大一小个 set，每次插入数据无脑往大的 set 里插入，插入之后做一个 balance 操作，所谓 balance 就是平均一下两个 set 的大小，然后把小 set 中的最大值 mx 和大 set 中的最小值 mi 对比一下，如果 mx > mi，那么就在两个 set 中交换一下两个元素；

这题的思路十分类似，也可以维护两个大小 set，set 的元素是一个 pair<int, int>, 表示某个数字 v <v 在滑动窗口内的次数，v>，同时为了知道 v 在滑动窗口内的次数，所以还需要一个计数器 map，保存 v 的次数，最后答案就是大 set 中数字的总和啦（次数*数字的累积和）；

具体思路如下：

定义 balance 操作逻辑

- 如果 large set 的 size 大于 x，那么就把最小元素往 small set 里塞
- 如果 large set 的 size 小于 x，并且 small set 不为空，那么就把 small set 的最大值往 large 里放；
- 最后比较 small set 的 最大值 和 large set 的最小值，如果满足之前说的要求，就交换一下；

算法完整解法：

- 先遍历前 k 个数，统计 map，然后把 pair 对全部插入大 set 中，ans 初始化为前 k 个数之和，balance 一下，得到第一个 ans 值；
- 然后从 k 下标开始遍历
  - 维护滑动窗口，添加当前值 a[i]: 首先要判断一下，当前新增的 a[i] 是否已经出现过，如果出现过，那么是在哪个 set，原来在哪个 set 就往哪个 set 里 erase 旧值，insert 新值 <次数+1， a[i]>，否则无脑往大 set 里插；
  - 删除 a[i-k]: 同理，原来在哪个 set 就从哪个 set 里 erase 旧值，insert <次数-1， a[i-k]>
  - 同时不要忘了维护计数器，mp[a[i]] ++, mp[a[i-1]] --， 还有操作 large set 操作的时候，同步维护一下 ans；
  - 单次遍历的最后 balance 一下，ans 值就是答案啦，push 到答案列表里。

代码如下，仅使用 c++ STL pair，set 和 unordered_map， 无任何特殊优化：

## Code

``` cpp
#include<iostream>
#include<bits/stdc++.h>
#include<limits>
using namespace std;
using pii=pair<int, int>;
class Solution {
public:
    long long ans;
    long long get(pii x){
        return x.first*1LL*x.second;
    }
    void balance(set<pii> &le, set<pii> &gt, int x){
        while(gt.size() > x){
            auto p = *gt.begin();
            le.insert(p);
            gt.erase(p);
            ans -= get(p);
        }
        while(le.size() > 0 && gt.size() < x){
            auto x = *le.rbegin();
            le.erase(x);
            gt.insert(x);
            ans += get(x);
        }
        while(le.size() > 0){
            auto x = *le.rbegin();
            auto y = *gt.begin();
            if (x < y){
                break;
            }else{
                le.erase(x);
                gt.erase(y);
                gt.insert(x);
                le.insert(y);
                ans += get(x) - get(y);
            }
        }
    }
    vector<long long> findXSum(vector<int>& a, int k, int x) {
        ans = 0;
        int n = a.size();
        unordered_map<int, int> mp;
        set<pii>  le, gt;
        for(int i=0; i<k; i++){
            mp[a[i]] ++;
            ans += a[i];
        }
        for(auto && p : mp){
            // cout<<p.first<<" "<<p.second<<"\n";
            gt.insert({p.second, p.first});
        }
        balance(le, gt, x);
        vector<long long> res{ans};
        for(int i=k; i<n; i++){
            int cnt = mp.count(a[i]) ? mp[a[i]] : 0;
            pii p{cnt, a[i]};
            if(gt.count(p)){
                gt.erase(p);
                gt.insert({cnt+1, a[i]});
                ans += a[i];
            }else if(le.count(p)){
                le.erase(p);
                le.insert({cnt+1, a[i]});
            } else {
                gt.insert({cnt+1, a[i]});
                ans += a[i];
            }
            mp[a[i]] ++;

            int v = a[i-k];
            cnt = mp.count(v) ? mp[v] : 0;
            p = pii{cnt, v};
            if(gt.count(p)){
                gt.erase(p);
                if(cnt > 1) gt.insert({cnt-1, v});
                ans -= v;
            }else{
                le.erase(p);
                if(cnt > 1) le.insert({cnt-1, v});
            }
            balance(le, gt, x);
            res.push_back(ans);
            mp[v] --;
        }
        return res;
    }
};

// int main()
// {
//     auto a = vector<int>{1,1,2,2,3,4,2,3};
//     auto res = Solution().findXSum(a, 6, 2);
//     for(auto x : res){
//         cout<<x<<" ";
//     }
//     cout<<"\n";
// }

```

很好理解吧~ 而且即使这样无脑插入写法性能也能打败 96%+ 哦，具体见：<https://leetcode.cn/problems/find-x-sum-of-all-k-long-subarrays-ii/submissions/678236110/?envType=daily-question&envId=2025-11-05>
