---
layout: post
title: Set Matrix Zeroes
date: '2022-06-07'
categories: arrayeasy
published: true
permalink: /:categories/:title 
---

### [Question Link : Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/)

-----------------------------------------------------------------------------------------------------------
&nbsp;
**✅ *Approach 1 : Using Brute force***

You can simply use containers to iterate through whole matrix and see for the zeroes in it. If you have any, change the row and the column containers's respective index values from -1 to 0, marking the presence of the 0 in that row and column.

In next iteration, you can check if the row and column containers have -1 in them. If yes, you can change the value of the corresponding element to 0.&nbsp;

```cpp
class Solution {
  void setZeroes(vector<vector<int>>& matrix){ 
    int R = matrix.size();
    int C = matrix[0].size();
    vector<int>rows(R,-1);
    vector<int>cols(C,-1);
    
    // Essentially, we mark the rows and columns that are to be made zero
    for (int i = 0; i < R; i++) {
      for (int j = 0; j < C; j++) {
        if (matrix[i][j] == 0) {
          rows[i] = 0;
          cols[j] = 0;
        }
      }
    }

    // Iterate over the array once again and using the rows and cols sets, update the elements.
    for (int i = 0; i < R; i++) {
      for (int j = 0; j < C; j++) {
        if (rows[i] = 0 || cols[j] = 0) {
          matrix[i][j] = 0;
        }
      }
    }
  }
};
```

&nbsp;
**⌛ Time Complexity**  : O(n^2)
**🚀 Space Complexity** : O(n)

-----------------------------------------------------------------------------------------------------------

**✅ *Approach 2 : Rather than using additional variables to keep track of rows and columns to be reset, we use the matrix itself as an indicator.***

The idea is that we can use the first cell of every row and column as a flag. This flag would determine whether a row or column has been set to zero. This means for every cell instead of going to M+N cells and setting it to zero we just set the flag in two cells.

```cpp
if cell[i][j] == 0 {
    cell[i][0] = 0
    cell[0][j] = 0
}
```

&nbsp;
These flags are used later to update the matrix. If the first cell of a row is set to zero this means the row should be marked zero. If the first cell of a column is set to zero this means the column should be marked zero.

***🤖 Algorithm***

1. We iterate over the matrix and we mark the first cell of a row `i` and first cell of a column `j`, if the condition in the pseudo code above is satisfied. i.e. if `cell[i][j] == 0`.
2. ***The first cell of row and column for the first row and first column is the same i.e. `cell[0][0]`. Hence, we use two additional variables to tell us if the first column had been marked or not***
3. Now, we iterate over the original matrix starting from second row and second column i.e. `matrix[1][1]` onwards. For every cell we check if the row `r` or column `c` had been marked earlier by checking the respective first row cell or first column cell. If any of them was marked, we set the value in the cell to 0.
4. We then check for the presence of the 0 values in the first row and first column by checking if those two additional variables that we defined earlier and update the results accordingly.

&nbsp;

```cpp
class Solution {
vector<vector<int>> setmatrixzero(vector<vector<int>> &v){

    // Flags to determine if the zeroth row and zeroth column consists of zero or not
    bool row0 = false , col0 = false;
    for(int i = 0; i < v.size(); i++){
        // Colunn 0 consists of zero
        if(v[i][0] == 0) col0 = true;
        for(int j = 0; j < v[0].size(); j++){
            // Row 0 consists of zero
            if(v[0][j] == 0) row0 = true;
            // Other then Zeroth row and zeroth column, if there are any occurence of the zero in the matrix, if there are
            // then we will mark their presence in the Zeroth row and the Zeroth column
            if(v[i][j] == 0){
                v[0][j] = 0;
                v[i][0] = 0;
            }
        }
    }

    for(int i = v.size() - 1 ; i >= 1 ; i--){
        for(int j = v[0].size() - 1 ; j >= 1 ; j--){
            if(v[i][0] == 0 || v[0][j] == 0) v[i][j] = 0;
        }
    }

    //If 0th row consists of the zero, that means all the elements in the 0th row will be equal to the zero
    if(row0){
        for(int i = 0; i < v[0].size() ; i++) v[0][i] = 0;
    }

    //If 0th column consists of the zero, that means all the elements in the 0th column will be equal to the zero
    if(col0){
        for(int i = 0; i < v.size(); i++)  v[i][0] = 0;
    }
    return v;
}
};
```

&nbsp;
**⌛ Time Complexity**  : O(n^2)
**🚀 Space Complexity** : O(n)

-----------------------------------------------------------------------------------------------------------
💻🐼💻 If there are any suggestions / questions / mistakes in my post, please do let me know by using the following email template: 👇

Email Id : vipulmaheshwari.deogarh@gmail.coms
Subject : Question / Your Name
Body : Feedback / Suggestion / Any other comments / chit-chat
