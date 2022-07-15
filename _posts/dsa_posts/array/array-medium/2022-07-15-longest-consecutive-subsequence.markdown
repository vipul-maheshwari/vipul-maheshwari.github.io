---
layout: post
title: Longest Consecutive Subsequence
date: "2022-07-15"
categories: array-medium
published: true
permalink: /:categories/:title 
---

### [Question Link : Longest Consecutive Subsequence](https://leetcode.com/problems/longest-consecutive-subsequence/)

-----------------------------------------------------------------------------------------------------------
&nbsp;
**✅ *Approach 1 : Brute force***

Sort the entire nums and check for the biggest count such that the nums[i] == nums[i-1] + 1; as it will take O(nlogn) time to sort and    O(n) time to iterate again.( not submissible).&nbsp;

```cpp
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        int max_count = 0;
        sort(nums.begin(), nums.end());
        for(int i = 0; i < nums.size(); i++)
        {
            int count = 1;
            while(i+1 < nums.size() && nums[i+1] == nums[i] + 1)
            {
                count++;
                i++;
            }
            max_count = max(max_count, count);
        }
        return max_count;
    }
};
```

&nbsp;
**⌛ Time Complexity**  : O(nlogn) + O(n)
**🚀 Space Complexity** : O(1)

-----------------------------------------------------------------------------------------------------------
&nbsp;
**✅ *Approach 2 : Using Hashing***

We will first push all are elements in the HashSet. Then we will run a for loop and check for any number(x) if it is the starting number of the consecutive sequence by checking if the HashSet contains (x-1) or not. If ‘x’ is the starting number of the consecutive sequence we will keep searching for the numbers y = x+1, x+2, x+3, ….. And stop at the first ‘y’ which is not present in the HashSet. Using this we can calculate the length of the longest consecutive subsequence.&nbsp;

```cpp
class Solution {
public:
int longestConsecutive(vector<int>& nums) {
 if(nums.size() <= 1) return nums.size();
 unordered_set<int>temp(nums.begin(), nums.end());
 int count_max = 0; int cur_count = 0;
 for(auto &i : temp){
  //Current count will always be equal to the 1 initally as single element will always be a consecutive sequence.
  cur_count = 1;
  if(temp.find(i-1) == temp.end()){
   int current_start = i;
   while(temp.find(current_start+1) != temp.end()){
    current_start++;
    cur_count++;
   }
  }
  count_max = max(count_max, cur_count);
 }
 return count_max;
}
};
```

&nbsp;
**⌛ Time Complexity**  : O(n)
**🚀 Space Complexity** : O(n)

-----------------------------------------------------------------------------------------------------------
💻🐼💻 If there are any suggestions / questions / mistakes in my post, please do let me know by using the following email template: 👇

Email Id : vipulmaheshwari.deogarh@gmail.com
Subject : Question / Your Name
Body : Feedback / Suggestion / Any other comments / chit-chat
