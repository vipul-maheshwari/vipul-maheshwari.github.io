---
layout: post
title: Pascal's Triangle 2
date: '2022-07-14'
categories: arraymedium
published: true
permalink: /:categories/:title
---

### [Pascal's Traiangle 2](https://leetcode.com/problems/pascals-triangle-ii)

-----------------------------------------------------------------------------------------------------------
&nbsp;
**β *Approach 1 : Using [Pascal's Triangle](https://vipul-maheshwari.github.io/arraymedium/Pascal's-Triangle)***

Create a Pascalβs triangle using Pascal's Triangle and then return that particular row associated with it. Eazzzy.&nbsp;

```cpp
class Solution {
public:
vector<vector<int>> generate(int numRows) {
 vector<vector<int>>res;
 for (int i = 0; i < numRows; i++) {
  //initializing a new row vector filled with i+1 number of 1s
  vector<int> row(i+1, 1);
  for (int j = 1; j < i; j++) {
   row[j] = res[i - 1][j] + res[i - 1][j - 1];
  }
  res.push_back(row);
 }
 return res;
}

vector<int> getRow(int rowIndex) {
 
 //generate a pascal's triangle of size rowIndex+1
 vector<vector<int>>triangle = generate(rowIndex+1);
 vector<int>row;

 for(int i = 0; i < triangle[rowIndex].size(); i++){
  row.push_back(triangle[rowIndex][i]);
 }
 return row;
}
};
```

&nbsp;
**β Time Complexity**  : O(n)
**π Space Complexity** : O(1)

-----------------------------------------------------------------------------------------------------------
π»πΌπ» If there are any suggestions / questions / mistakes in my post, please do let me know by using the following email template: π

Email Id : vipulmaheshwari.deogarh@gmail.com
Subject : Question / Your Name
Body : Feedback / Suggestion / Any other comments / chit-chat
