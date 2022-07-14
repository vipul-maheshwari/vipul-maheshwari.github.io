---
layout: post
title: Print Subsequences
date: '2022-07-07'
categories: dpmedium
published: true
permalink: /:categories/:title 
---

### [Question Link : Print Subsequences](https://www.geeksforgeeks.org/generating-all-possible-subsequences-using-recursion/)

-----------------------------------------------------------------------------------------------------------
&nbsp;
**✅ *Approach 1 : Using recursion***

Most people get confused in subarrays and subsequences. The difference is simple though, subarrays are contiguous collections of an array, while subsequences need not to be contiguous. For an example, if the given array is [3,2,1] then subarrays are [3,2,1] , [3,2] , [3] , [2,1] , [2] , [1] , [] , subsequences are [3,2,1] , [3,2] , [3,1] , [3] , [2,1] , [2] , [1] , []. The number of subsequences is 8. The number of subarrays is 7. You can easily find the subarrays using two nested loops, for each element of the array, you can generate all the subarrays of the remaining elements. But for subsequences, you can't generate all the subsequences with the same logic.&nbsp;

***🤖 Algorithm to generate all subsequences of the string using recursion***

- For every character in the string, we have two options, either we can take that character in our subsequence or we can skip that character. So we will start with the first character of the string and recursively call the function for the remaining characters for the two options.
- If we choose to take the character, then we will add the character to the subsequence and call the function for the remaining characters.
- If we choose to skip the character, then we will skip the character and call the function for the remaining characters.
🔴 Note: If we choose to skip the character, make sure you remove that character from the original string and then call the function for the remaining characters.

&nbsp;

```cpp
void helper(int index, vector<string>&res, string &s, string temp_string){
//if index of the given temp string is equal to the string s then we know we reached to the base case
 if(index == s.size()){
  res.push_back(temp_string);
  return;
 }

 //taking the current character literal
 temp_string.push_back(s[index]);

 //recursive call for the rest of the elements
 helper(index+1,res,s,temp_string);

 //other option, don't take the current character and then find the subsequence
 temp_string.pop_back();

 //recursive call for the rest of the elements
 helper(index+1,res,s,temp_string);
 return;
}

//Main function
vector<string> generateallsubsequnces(string s){
 vector<string>res;
 string s_temp = "";
 helper(0,res,s,s_temp);
 return res;
}
```

&nbsp;
**⌛ Time Complexity**  : O(2^n)
**🚀 Space Complexity** : O(n) (recursive stack)

-----------------------------------------------------------------------------------------------------------

**✅ *Approach 2 : Using Bit operations***

For a string of given length n, there are 2^n possible subsequences. We can use bit operation to find all the subsequences. What we have to do is for every character in the string, assume it occupies a bit in the bitmask. Starting with the first bit, we will check if the current bit is 1 or not, if it is 1 include that character in the current subsequnces and if not, skip that character. Do the same for all the 2^n values starting from the 0 to 2^n-1. &nbsp;

```cpp
vector<string> generateallsubsequnces(string s){
    vector<string>res;
    int size_string = s.size();
    int subsequence_count = pow(2,size_string);

    for(int i = 0; i < subsequence_count; i++){
        string temp_string = "";
        for(int j = 0; j < size_string; j++){
            if((i & (1<<j))){
                temp_string.push_back(s[j]);
            }
        }
    res.push_back(temp_string);
    }
    return res;
}
```

&nbsp;
**⌛ Time Complexity**  : O(2^n) + O(n)
**🚀 Space Complexity** : O(2^n)

-----------------------------------------------------------------------------------------------------------
💻🐼💻 If there are any suggestions / questions / mistakes in my post, please do let me know by using the following email template: 👇

Email Id : vipulmaheshwari09042001@gmail.com
Subject : Question / Your Name
Body : Feedback / Suggestion / Any other comments / chit-chat
