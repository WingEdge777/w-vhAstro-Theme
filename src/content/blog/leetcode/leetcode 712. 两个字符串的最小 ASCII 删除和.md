---
title: "leetcode 712. 两个字符串的最小 ASCII 删除和 (medium)"
categories: "leetcode"
tags: ["公共子序列", "动态规划"]
id: "fc1c09c2eb681651"
date: 2026-01-10 12:13:13
cover: "/assets/images/banner/6c055e6953b0b598.webp"
---

:::note
经典动态规划-公共子序列问题的基础变形题
:::

## 题面

给定两个字符串 s1 和 s2，返回使两个字符串相等所需删除字符的 ASCII 值的最小和 。

### 解析

经典的公共子序列问题是找出两个序列的最长公共子序列（可以通过删除或增加字符），动态规划主要思路是找到状态的定义以及状态间转移方式。
公共子序列需要维护长度，该题相对应的要维护的状态变成了 被删除的 ASCII 和。

因此我们用 dp[i][j] 表示 字符 s1[0...i] 和 s2[0...j] 变为相同（公共）子序列所需要删除的字符 ASCII 和。

显然，转移方程有：

- dp[i][j] = min(dp[i-1][j] + s1[i], dp[i][j-1] + s2[j])
- 特别的，当 s1[i] == s2[j] 时，dp[i][j] 还可以直接由 dp[i-1][j-1] 而来，因此三者中取最小值

### ac 代码如下

```cpp
int dp[1010][1010];
class Solution {
public:
    int minimumDeleteSum(string s1, string s2) {
        dp[0][0] = 0;
        int n = s1.length(), m = s2.length();
        for (int i = 1; i <= n; i++) dp[i][0] = dp[i - 1][0] + s1[i - 1];
        for (int j = 1; j <= m; j++) dp[0][j] = dp[0][j - 1] + s2[j - 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                dp[i][j] = min(dp[i - 1][j] + s1[i - 1], dp[i][j - 1] + s2[j - 1]);
                if (s1[i - 1] == s2[j - 1]) {
                    dp[i][j] = min(dp[i][j], dp[i - 1][j - 1]);
                }
            }
        }
        return dp[n][m];
    }
};
``
看，代码几乎和公共子序列问题一样。所以，对于动态规划问题首先还是要理解答案的状态定义，自然而然就能举一反三。
这份代码无任何特殊优化，可以打败 91% 的提交，具体见：https://leetcode.cn/problems/minimum-ascii-delete-sum-for-two-strings/submissions/690312584
