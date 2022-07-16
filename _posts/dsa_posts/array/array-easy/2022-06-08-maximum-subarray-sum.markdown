---
layout: post
title: Maxmimum Subarray Sum
date: '2022-06-08'
categories: arrayeasy
published: true
permalink: /:categories/:title 
---

### [Maxmimum Subarray Sum](https://leetcode.com/problems/maximum-subarray/)

-----------------------------------------------------------------------------------------------------------

&nbsp;
**✅ *Approach 1 : Using Brute force***

You can easily solve this problem by iterating over the array two times. First iteration is for considering every subarray and second iteration is for finding the maximum sum among all subarrays. A very niche approach.&nbsp;

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int max_sum = INT_MIN;
        for(int i = 0; i < nums.size(); i++){
            int sum = 0;
            for(int j = i; j < nums.size(); j++){
                sum += nums[j];
                max_sum = max(max_sum, sum);
            }
        }
        return max_sum;
    }
};
```

&nbsp;
**⌛ Time Complexity**  : O(n^2)
**🚀 Space Complexity** : O(1)

-----------------------------------------------------------------------------------------------------------
&nbsp;
**✅ *Approach 2 : Using Kadane's Algorithm***

There is a simple algorithm to solve this problem. What you can do is, using only one pass, we can find the maximum sum subarray. For every element in the array, we have two options, either include that element in the current subarray, or not include that element in the current subarray. For an example if the array is [-1,-2,4] then element 4 better not to include in the current subarray of -1 and -2, because it will give a sum of -1, which is less than the 4 itself, so the 4 singleton subarray is the best choice as of now.

Take the maximum sum between the current subarray and the maximum subarray till now.

***🤖 Algorithm***

1. Initialize the current sum as 0.
2. For every element in the array, do the following:
    a. Current Sum = maximum(Current Sum + arr[i], arr[i])
    b. Maximum Sum = maximum(Maximum Sum, Current Sum)
3. Return Maximum Sum.&nbsp;

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int max_sum = INT_MIN;
        int curr_sum = 0;
        for(int i = 0; i < nums.size(); i++){
            curr_sum = max(curr_sum + nums[i], nums[i]);
            max_sum = max(max_sum, curr_sum);
        }
        return max_sum;
    }
};
```

&nbsp;
**⌛ Time Complexity**  : O(n)
**🚀 Space Complexity** : O(1)

-----------------------------------------------------------------------------------------------------------
&nbsp;
**✅ *Approach 3 : Follow up → How to use the divide and conquer strategy to solve the same problem?***

This involves recreating the problem by looking at the maximum subarray sum in such a way that it lies somewhere in between three different array choices-

- entirely in the left-half of the array `[L, mid-1]`, OR
- entirely in the right-half of the array `[mid+1, R]`, OR
- In array consisting of mid element along with some part of left-half and some part of right-half such that these very form contiguous subarray - `[L', R'] = [L', mid-1] + [mid] + [mid+1,R']`, where `L' >= L` and `R' <= R`

As, we can recursively divide the array into sub-problems on the left and right halves and then combine these results on the way back up to (bottom up) find the maximum subarray sum.

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        return maxSubArray(nums, 0, size(nums)-1);
    }
    int maxSubArray(vector<int>& A, int L, int R){
        if(L > R) return INT_MIN;
        int mid = (L + R) / 2, leftSum = 0, rightSum = 0;
        // leftSum = max subarray sum in [L, mid-1] and starting from mid-1
        for(int i = mid-1, curSum = 0; i >= L; i--)
            curSum += A[i],
            leftSum=max(leftSum, curSum);
        // rightSum = max subarray sum in [mid+1, R] and starting from mid+1
        for(int i = mid+1, curSum = 0; i <= R; i++)
            curSum += A[i],
            rightSum = max(rightSum, curSum);        
            // return max of 3 cases above
        return max({ maxSubArray(A, L, mid-1), maxSubArray(A, mid+1, R), leftSum + A[mid] + rightSum });
    }
};
```

&nbsp;
**⌛ Time Complexity**  : O(NlogN) , the recurrence relation can be written as T(N) = 2T(N/2) + O(N) since we are recurring for left and right half 2T(N/2) and also calculating maximal subarray including mid element for both the leftSum and the rightSum which takes O(N) to calculate. Solving this recurrence using master theorem, we can get the time complexity as O(NlogN)
**🚀 Space Complexity** : O(logN) as we are dividing the whole array into equivalent halves.

-----------------------------------------------------------------------------------------------------------

**✅ *Approach 4 : Can we optimize our divide and conquer strategy any further? Yes we can***

We can further optimize the previous solution further. The `O(N)` term in the recurrence relation of previous solution was due to computation of max sum subarray involving `nums[mid]` in each of the recursion. By looking carefully we can reduce that term to `O(1)` if we precompute it. This can be done by precomputing two arrays `pre` and `suf` where `pre[i]` will denote maximum sum subarray ending at `i` and `suf[i]` denotes the maximum subarray starting at `i`. `pre` is similar to `dp` array that we computed in dynamic programming solutions and `suf` can be calculated in similar way, just by starting iteration from the end.&nbsp;

![image](/assets_for_posts/dsa/array/photo_2022-07-12_19-05-48.jpg)

```cpp
class Solution {
public:
    vector<int> pre, suf;
    int maxSubArray(vector<int>& nums) {
        pre = suf = nums;
        for(int i = 1; i < size(nums); i++)  pre[i] += max(0, pre[i-1]);
        for(int i = size(nums)-2; ~i; i--)   suf[i] += max(0, suf[i+1]);
        return maxSubArray(nums, 0, size(nums)-1);
    }
    int maxSubArray(vector<int>& A, int L, int R){
        if(L >= R) return A[L];
        int mid = (L + R) / 2;
        return max({ maxSubArray(A, L, mid), maxSubArray(A, mid+1, R), pre[mid] + suf[mid+1] });
    }
};
```

&nbsp;
**⌛ Time Complexity**  : O(N) , the recurrence relation can be written as T(N) = 2T(N/2) + O(1) since we are recurring for left and right half 2T(N/2and calculating maximal subarray including mid element in O(1). Solving this recurrence using master theorem, we can get the time complexity as O(N)
**🚀 Space Complexity** : O(n) required by sufand pre .

**🔴 Note:**
The above divide and conquer solution works in O(N) but is once you have calculated pre and suf, we don’t even need to go for divide and conquer strategy? Actually we don’t, as we can easily calculate the solution by just considering the prefix maximum sum at each index value.

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        vector<int> pre = nums;
        for(int i = 1; i < size(nums); i++) pre[i] += max(0, pre[i-1]);
        return *max_element(begin(pre), end(pre));
    }
};
```

-----------------------------------------------------------------------------------------------------------
💻🐼💻 If there are any suggestions / questions / mistakes in my post, please do let me know by using the following email template: 👇

Email Id : vipulmaheshwari.deogarh@gmail.com
Subject : Question / Name
Body : Feedback / Suggestion / Any other comments / chit-chat
