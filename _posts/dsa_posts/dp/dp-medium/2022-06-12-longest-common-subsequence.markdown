---
layout: post
title: Longest Common Subsequence
date: '2022-06-12'
categories: dpmedium
published: true
permalink: /:categories/:title
---

### [Question Link : Longest Common Subsequence](https://www.leetcode.com/problems/longest-common-subsequence/) -- [Solution](https://youtu.be/Ua0GhsJSlWM)

--------------------------------------------------------------------------------

&nbsp;
**✅ *Approach 1 : Using Brute force***

The Longest Common Subsequence (LCS) problem is finding the longest subsequence present for the two given sequences in particularly same order i.e., find the longest sequence which can be obtained from the first original sequence by deleting some items and from the second original sequence by deleting other items. As in subsequence case, they are not necessarily required to occupy the consecutive positions with the original string

Consider this example:

```cpp
X: ABCBDAB
Y: BDCABA
 
The length of the LCS is 4
LCS are BDAB, BCAB, BCBA
```

&nbsp;
A naïve solution is to test if every subsequence of X[1…m] is likewise a subsequence of Y[1…n]. since there are 2^m feasible subsequences of X, the time complexity of this answer might be O(n.2^m), where m is the length of the primary array and n is the length of the second array.

The LCS problem has an optimal sub problem style this means that the problem may be broken down into smaller, simple "sub-problems" and simpler subproblems, and so on...

**🙋‍♀️ *for better understanding consider this example***

1.Let’s consider two sequences, X and Y, of length m and n that both end in the same element.

To find their LCS, shorten each sequence by removing the last element, find the LCS of the shortened sequences, and that LCS append the removed element. So, we can say that.

### LCS(X[1…m], Y[1…n]) = LCS(X[1…m-1], Y[1…n-1]) + X[m]    if X[m] = Y[n]

&nbsp;
2.Now suppose that the two sequences does not end in the same symbol that is now we need to decide if we want to take care of the LCS between (X[1…m], Y[1…n-1]) or (X[1…m-1], Y[1…n]) as we will take one of them whose LCS value is more than the other one.

Then the LCS of X and Y is the longer of the two sequences LCS(X[1…m-1], Y[1…n]) and LCS(X[1…m], Y[1…n-1]). To understand this property, let’s consider the two following sequences:

`X: ABCBDAB (n elements) Y: BDCABA (m elements)`

The LCS of these two sequences either ends with B (the last element of the sequence X) or does not.

**Case 1: If LCS ends with B, then it cannot end with A, and we can remove A from the sequence Y, and the problem reduces to LCS(X[1…m], Y[1…n-1]).**

**Case 2: If LCS does not end with B, then we can remove B from sequence X and the problem reduces to LCS(X[1…m-1], Y[1…n]). For example,**

`LCS(ABCBDAB, BDCABA) = maximum (LCS(ABCBDA, BDCABA), LCS(ABCBDAB, BDCAB))`
`LCS(ABCBDA, BDCABA) = LCS(ABCBD, BDCAB) + A`
`LCS(ABCBDAB, BDCAB) = LCS(ABCBDA, BDCA) + B`
`LCS(ABCBD, BDCAB) = maximum (LCS(ABCB, BDCAB), LCS(ABCBD, BDCA))`
`LCS(ABCBDA, BDCA) = LCS(ABCBD, BDC) + A`

&nbsp;
And so on…

```cpp
// Function to find the length of the longest common subsequence of
// sequences `X[0…m-1]` and `Y[0…n-1]`
int LCSLength(string X, string Y, int m, int n)
{
    // return if the end of either sequence is reached
    if (m == 0 || n == 0) {
        return 0;
    }
 
    // if the last character of `X` and `Y` matches
    if (X[m - 1] == Y[n - 1]) {
        return LCSLength(X, Y, m - 1, n - 1) + 1;
    }
 
    // otherwise, if the last character of `X` and `Y` don't match
    return max(LCSLength(X, Y, m, n - 1), LCSLength(X, Y, m - 1, n));
}
```

&nbsp;
**⌛ Time Complexity** : O(2^m.2^n)
**🚀 Space Complexity** : O(n)

⭐ The worst-case time complexity of the above solution is O(2(m+n)) and occupies space in the call stack, where m and n are the length of the strings X and Y. The worst case happens when there is no common subsequence present in X and Y (i.e., LCS is 0), and each recursive call will end up in two recursive calls.

--------------------------------------------------------------------------------

**✅ *Approach 2 : Using DP.***

The LCS problem exhibits overlapping subproblems. A problem is said to have overlapping subproblems if the recursive algorithm for the problem solves the same subproblem repeatedly rather than generating new subproblems.

Let’s consider the recursion tree for two sequences of length 6 and 8 whose LCS is 0.

![LCSTREE](/assets_for_posts/dsa/dp/LCS/LCS-Recursion-Tree.png)

As we can see, the same subproblems (highlighted in the same color) are getting computed repeatedly. We know that problems having optimal substructure and overlapping subproblems can be solved by dynamic programming, in which subproblem solutions are memoized rather than computed repeatedly. This method is demonstrated below in C++:

```cpp
class Solution {
public:
int longestCommonSubsequence(string text1, string text2) {

    //Creating a 2D vector to store the longest common subsequence at each position
    vector<vector<int>>dp(text1.size()+1, vector<int>(text2.size()+1));

    for(int i = text1.size() - 1; i >= 0; i--) {
        for(int j = text2.size() - 1; j >= 0; j--) {
            if(text1[i] == text2[j]){
                dp[i][j] = 1 + dp[i+1][j+1];
            }

            if(text1[i] != text2[j]){
                dp[i][j] = max(dp[i+1][j], dp[i][j+1]);
            }
        }
    }
    return dp[0][0];
}
};
```

&nbsp;
**⌛ Time Complexity** : O(n.m) where n and m are the length of the strings text1 and text2.
**🚀 Space Complexity** : O(n.m)

--------------------------------------------------------------------------------
📕References

- 📑 [Article](https://www.techiedelight.com/longest-common-subsequence/)

--------------------------------------------------------------------------------
💻🐼💻 If there are any suggestions / questions / mistakes in my post, please do let me know by using the following email template: 👇

Email Id : vipulmaheshwari.deogarh@gmail.com
Subject : Question / Your Name
Body : Feedback / Suggestion / Any other comments / chit-chat
