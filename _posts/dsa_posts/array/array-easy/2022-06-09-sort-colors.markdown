---
layout: post
title: Dutch National flag algorithm 
date: '2022-06-08'
categories: arrayeasy
published: true
permalink: /:categories/:title 
---

### [Question Link : Sort Colors](https://leetcode.com/problems/sort-colors/)

-----------------------------------------------------------------------------------------------------------
&nbsp;
**✅ *Approach 1 : Using STL***
You can easily solve this problem by using sort() function from STL. This is a very simple and easy to understand approach, but it is clearly mentioned in the documentation of the question that we are not allowed to use the sort() or any kind of STL function.
&nbsp;

```cpp
class Solution {
public:
    void sortColors(vector<int>& nums) {
        sort(nums.begin(), nums.end());
    }
};
```

&nbsp;
**⌛ Time Complexity**  : O(n*log n)
**🚀 Space Complexity** : O(1)

-----------------------------------------------------------------------------------------------------------
&nbsp;
**✅ *Approach 2 : Count the number of 0's, 1's, and 2's'***

Count the number of 0's, 1's, and 2's in the array. Then, use the number of 0's to place 0's in the beginning of the array, 1's in the middle, and 2's in the end. &nbsp;

```cpp
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int n0 = 0, n1 = 0, n2 = 0;
        for(int i = 0; i < nums.size(); i++){
            if(nums[i] == 0)
                n0++;
            else if(nums[i] == 1)
                n1++;
            else
                n2++;
        }
        for(int i = 0; i < n0; i++)
            nums[i] = 0;
        for(int i = n0; i < n0 + n1; i++)
            nums[i] = 1;
        for(int i = n0 + n1; i < nums.size(); i++)
            nums[i] = 2;
    }
};
```

&nbsp;
**⌛ Time Complexity**  : O(n)
**🚀 Space Complexity** : O(1)

-----------------------------------------------------------------------------------------------------------
&nbsp;
**✅ *Approach 3 : Using Dutch National Flag Algorithm***

In this algorithm, we will use three pointers *left*, *right*, and *mid* to validate the presence of 0, 1, and 2 in the array. *left* will point to the first element in the array, *right* will point to the last element in the array, and *mid* will point to the middle element in the array.

***🤖 Algorithm***
1.Initialize *left* to 0, *right* to nums.size() - 1, and *mid* to 0
2.While *mid* is less than or equal to *right*:
    1.If *nums[mid]* is 0, swap *nums[mid]* with *nums[left]*, increment *left*, and increment *mid*.
    2.Else if *nums[mid]* is 1, increment *mid*.
    3.Else, swap *nums[mid]* with *nums[right]*, decrement *right*.
3.Return

```cpp
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int left = 0, right = nums.size() - 1, mid = 0;
        while(mid <= right){
            if(nums[mid] == 0){
                swap(nums[mid], nums[left]);
                left++;
                mid++;
            }
            else if(nums[mid] == 1){
                mid++;
            }
            else{
                swap(nums[mid], nums[right]);
                right--;
            }
        }
    }
};
```

&nbsp;
**⌛ Time Complexity**  : O(n)
**🚀 Space Complexity** : O(1)

-----------------------------------------------------------------------------------------------------------
💻🐼💻 If there are any suggestions / questions / mistakes in my post, please do let me know by using the following email template: 👇

Email Id : vipulmaheshwari.deogarh@gmail.com
Subject : Question / Name
Body : Feedback / Suggestion / Any other comments / chit-chat
