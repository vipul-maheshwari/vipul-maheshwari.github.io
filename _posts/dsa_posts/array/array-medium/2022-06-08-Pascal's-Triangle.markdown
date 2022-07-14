---
layout: post
title: Pascal's Triangle
date: '2022-06-08'
categories: arraymedium
published: true
permalink: /:categories/:title
---

### [Question Link : Pascal's Triangle](https://leetcode.com/problems/pascals-triangle/)

-----------------------------------------------------------------------------------------------------------

&nbsp;
**✅ *Approach 1 : Dynamic Programming - Iterative version***

The Pascal Triangle problem has a very simple Dynamic programming approach. Every element of the row is shaped by considering the sum of two numbers which are placed just immediately above them in the previous row.  So using the concept of DP, we only need to look out for the elements in the previous row of the triangle to calculate the new rows.

Initialize the number of columns for each row and set all it’s elements as 1. This will make our lives easier as we only need to change the innermost elements leaving the edges as 1. And then change the inner elements based on the sum of it’s two previous elements in it's immediate previous row.

```cpp
class Solution {
//Build an array with ans[i] = nums[nums[i]]
vector<vector<int>> generate(int numRows) {
    vector<vector<int>> pascaltri;
    for (int i = 0; i < numRows; i++) {
        //Setting all the elements of the rows as 1
        vector<int> rownum(i + 1, 1);
        for (int j = 1; j < i; j++) {
            //Transforming the current elements of the Pascal Triangle
            rownum[j] = pascaltri[i - 1][j] + pascaltri[i - 1][j - 1];
        }
        //Pushing the newly generated row of the pascal traingle into the the matrix
        pascaltri.push_back(rownum);
    }
    return pascaltri;
}
};
```

&nbsp;
**⌛ Time Complexity**  : O(n^2)
**🚀 Space Complexity** : O(n^2)

-----------------------------------------------------------------------------------------------------------

**✅ *Approach 2 : Dynamic Programming - Recursive version***

Logic for the recursive version is same as what we did in the iterative version. Instead, we are using Top-Down approach where we go from the bottom to the top until the top row is not generated and then we are coming back to the bottom again to form the new rows using the previously generated rows.

```cpp
class Solution {
vector<vector<int>>ans;
public:    
vector<vector<int>>& generate(int n) {
    if(n) {
        generate(n-1);                       // generate above row first
        ans.push_back(vector<int>(n,1));  // insert current row into triangle
        for(int i = 1; i < n-1; i++)         // update current row values using above row
            ans[n-1][i] = ans[n-2][i] + ans[n-2][i-1];    
    }
    return ans;
}
};
```

&nbsp;
**⌛ Time Complexity**  : O(n^2)
**🚀 Space Complexity** : O(n^2)

-----------------------------------------------------------------------------------------------------------
💻🐼💻 If there are any suggestions / questions / mistakes in my post, please do let me know by using the following email template: 👇

Email Id : vipulmaheshwari.deogarh@gmail.com
Subject : Question / Your Name
Body : Feedback / Suggestion / Any other comments / chit-chat
