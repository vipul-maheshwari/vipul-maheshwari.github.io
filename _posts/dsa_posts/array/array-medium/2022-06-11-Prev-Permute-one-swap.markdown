---
layout: post
title: Prev Permute one swap
date: '2022-06-12'
categories: arraymedium
published: true
permalink: /:categories/:title
---

### [Question Link : Prev Permute one swap](https://leetcode.com/problems/previous-permutation-with-one-swap/)

-----------------------------------------------------------------------------------------------------------

&nbsp;
**✅ *Approach 1 : Using Brute force approach (Best possible way to solve the problem);***

**NOTE: PLEASE READ THIS WHOLE PARA WITH ATMOST CONCENTRATION**
When I first saw this problem, I was confused. I thought it was a problem of finding the previous permutation of a given vector. But it turns out that it is a problem of finding the previous permutation of a given vector with **one swap**.

To understand this problem, let's look at the following example.
Let's say our vector is [3,1,1,3], if you find the previous permutation of the given vector, you will get [1,1,3,3]. But if you find the previous permutation of the given vector with one swap, you will get [1,3,1,3]. So what's gone wrong?

The problem is that we have to find the previous permutation of the given vector with one swap. But we can't find the previous permutation of the given vector with one swap using the same algorithm we used to find the [previous permutation](https://vipul-maheshwari.github.io/dsa/Previous-Permutation).

Back to the same example, let's say we need to find the previous permutation of the given vector using the same algorithm as we used to find the previous permutation of the given vector.

So we need to find the deeps of the given vector. So starting with the last element, as soon as we find a element that is smaller than the element on its left, we know that we have found the deeps of the given vector. So for this case, the lower element is 1 and upper element is 3.

Next, we need to perform the swapping operation, so we need to swap the lowest possible element ( greater than 1 if there is any) with 3. Starting from the back of the grid, first comes 3, as 3 is not less than than our 1st index 3 so we continue, now we find the element 1, now there are two choices with us:

1.Swap this 1 with 3, we get [1,1,3,3] but this permutation is not the previous permutation of [3,1,1,3], to find the immediate previous permutation, we need to reverse the subarray starting from the Deep index (index 1) to last index (index 3) which will give us [1,3,3,1] but this is not the right answer as this answer involves more than one swap.

2.Continue with the backward iteration

When we choose the second choice, we reach to the original 1st index 1, as now we have no where to go, we will swap this 1 with 3 and we get [1,3,1,3]. But when you look out closely, we avoided swapping 3 with 1(at index 2) that is we consider the dupicates case, if we swap the dupicate number of deep present at less significant position compared to the deep's position, we will not get the immediate previous permutation and as we want to find the immediate previous permutation, but also utilizing the abilites of only one swap, then that's the best choice for us. For example if the vector was [3,1,1,1,3] then if we swap 0th index 3 with the 3rd index 1 then we would get [1,1,1,3,3] and as we are only allowed to do one swap this would be our current answer, but the best possible solution would be avoiding the duplicates of the deep element and swap the 3 at 0th index with the deep element only which will give us [1,3,1,1,3] which is more nearer previous permutation of [3,1,1,1,3] rather than [1,1,1,3,3].

Now the fun fact is, when I was solving this question, I thought we need to find the immediate previous permutation of the given vector that is I need to follow the [Previous Permutation algorithm](https://vipul-maheshwari.github.io/dsa/Previous-Permutation) but doing so, I will reverse the elements from the deep to the last of the array which will take more than one swap (obviously, as we have already performed one swapping), and doing so the result becomes [1,3,3,1] but this is wrong.

As we can see, the result is [1,3,1,3] and this is the correct answer for the previous permutation of [3,1,1,3] with only one swap.

**follow up question? : Why we find the swapping element from the back of the array?** : This is because as soon as we find our deep, swapping that deep element with the higer element on it's left side will give us previous permutation, but in case if we need to find the immediate previous permutation, we need to look out for the immediate higer element (X) whose value is smaller than (Y, element on the left side of the deep) such that (D, our deep element is smaller than X). That is we need to find the least significant element in the array that is greater than the deep element (if there is any) and whose value is smaller than the Y. (Y > D < X)

For example, if the array is [6,3,4,5,8] then if we replace 6 with 5 then we get [5,3,4,6,8] -> reverse -> [5,8,6,4,3] where as we we replace 6 with 4 then we get [4,3,6,5,8] -> reverse -> [4,8,6,5,3] and if we replace 6 with 3 then we get [3,6,4,5,8] -> reverse -> [3,8,6,5,4]. Comparing all three, we can conclude that [5,8,6,4,3] is most previous permutation rather than [4,8,6,5,3] and [3,8,6,5,4] because the value of 5(X) is greater than D(3) and smaller than 6(Y)

I wrote everything I could, if you didn't get the idea, please read the paragraph again, and if you still don't get the idea, please let me know by dropping an email.

```cpp
class Solution {
public:
vector<int> prevPermOpt1(vector<int> &nums){

  int index = nums.size() - 1;
  
  while(index > 0){

    if(nums[index] < nums[index-1]){
      break;
    }
    index--;
  }

  if(index > 0){
    
    int right = nums.size()-1;
    while(right >= index){
      
      //nums[right] == nums[right-1] -> To handle the duplicates case i.e [6,3,4,4,4,4,8], the best possible solution would be swapping
      //the 6 with the 4 at index 2

      if(nums[right] == nums[right-1] || nums[right] >= nums[index-1]){
        right--;
      }

      //As soon as you get the rightous element, you can swap it with the higher value element on the left side of the deep element
      else{
        swap(nums[right], nums[index-1]);
        break;
      }  
    }
  }
  return nums;
}
};
```

&nbsp;
**⌛ Time Complexity** : O(n+m) where m is less than n
**🚀 Space Complexity** : O(1)

-----------------------------------------------------------------------------------------------------------
💻🐼💻 If there are any suggestions / questions / mistakes in my post, please do let me know by using the following email template: 👇

Email Id : vipulmaheshwari09042001@gmail.com
Subject : Question / Your Name
Body : Feedback / Suggestion / Any other comments / chit-chat
