---
layout: post
title: Next Permutation
date: '2022-06-07'
categories: arraymedium
published: true
permalink: /:categories/:title 
---

### [Question Link : Next Permutation](https://leetcode.com/problems/next-permutation/)

-----------------------------------------------------------------------------------------------------------
&nbsp;
**✅ *Approach 1 : Using STL in-built function : bool next_permutation (BidirectionalIterator first,BidirectionalIterator last);***

You can simply use STL's inbuilt function [next_permutation()](https://www.geeksforgeeks.org/stdnext_permutation-prev_permutation-c/) to get the next permutation of the given array.
&nbsp;

```cpp
class Solution {
    vector<int> nextPermutation(vector<int>&v){
        next_permutation(v.begin(), v.end());
        return v;
    }
};
```

&nbsp;
Since there are n! permutations for a vector of length n, and each permutation takes O(n) time, the time complexity of the above solution is O(n * n!). The best case happens when the string contains all repeated characters, and the worst-case happens when the string contains all distinct elements.

[next_permutation implementation](https://www.techiedelight.com/std_next_permutation-overview-implementation/)

**⌛ Time Complexity**  : O(n^2)
**🚀 Space Complexity** : O(n)

-----------------------------------------------------------------------------------------------------------
&nbsp;
**✅ *Approach 2 : Using the intution behind the lexiographical order : Finding the peaks and swapping them to generate the next higher order sequence***

You can generate the next permutation by finding the peaks of the given vector and swapping them with the smaller element on the left side of the peak. To get the better understanding, see this [YouTube](https://youtu.be/6qXO72FkqwM) video.

Well, this is more like a math problem and there exists a decisive algorithm to solve this.

***🤖 Algorithm***
Step 1: Find the largest index k, such that A[k] > A[k-1] (Peak). If no such index exists, the permutation i s the last permutation. Just reverse the vector and return the result.
Step 2: Find the largest index l, such that A[l] > A[k-1].
Step 3: Swap A[k-1] and A[l].
Step 4: Sort the elements in the subarray A[k]-A[n-1].
Step 5: Return the result.

```cpp
class Solution {
vector<int> nextPermutation(vector<int>& v) {

    int index = v.size()-1;
    //finding the rightmost peak from the given sequence
    while(index > 0){
        if(v[index - 1] < v[index]){
            break;
        }
        index--;
    }

    if(index > 0){
        int rightmost = v.size()-1;
        //We want to find the element whose value is greater than that of the left element of the peak
        while(v[index-1] >= v[rightmost]){
            rightmost--;
        }
        swap(v[index-1], v[rightmost]);
        sort(v.begin() + index, v.end());
    }

    //Else condition only arises when the sequence is sorted in non-increasing way
    else{
        reverse(v.begin(), v.end()); 
    }
    return v;
}
};
```

&nbsp;
Time complexity is mixed up function of O(n) and O(nlog(n)) values. The first part of the algorithm is O(n) where we iterate our vector 2 times in a row and the second part is O(nlog(n)) which is the sorting and reversing part.

**⌛ Time Complexity** : O(n + log(m)) where m is the number of elements from the peak to the last element of the vector.
**🚀 Space Complexity** : O(1)

-----------------------------------------------------------------------------------------------------------
💻🐼💻 If there are any suggestions / questions / mistakes in my post, please do let me know by using the following email template: 👇

Email Id : vipulmaheshwari.deogarh@gmail.com
Subject : Question / Your Name
Body : Feedback / Suggestion / Any other comments / chit-chat
