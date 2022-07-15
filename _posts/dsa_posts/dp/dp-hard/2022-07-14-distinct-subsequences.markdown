---
layout: post
title: Distinct Subsequences
date: "2022-07-14"
categories: dphard
published: true
permalink: /:categories/:title
---

### [Question Link : Distinct Subsequencs](https://leetcode.com/problems/distinct-subsequences/)

-----------------------------------------------------------------------------------------------------------
&nbsp;
**✅ *Approach 1 : Using Recursion***

The basic and naïve approach would be find all the subsequences of the string s, count the number of occurrence of each one of them and check if there is any subsequence equal to the string t, if there is, increase the count equal to the number of occurrence of that subsequence. &nbsp;

```cpp
class Solution {
public:
//Helper function
void helper(int index, vector<string>&res, string &s, string temp_string){

 //if index of the given temp string is equal to the string s then we know we reached to the base case
 if(index == s.size()){
  res.push_back(temp_string);
  return;
 }

 //taking the current character literal
 temp_string.push_back(s[index]);
 helper(index+1,res,s,temp_string);
 temp_string.pop_back();
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


int numDistinct(string s, string t) {
 vector<string>s_subsequences = generateallsubsequnces(s);
 
 //mapping unique subsequnces of s
 unordered_map<string,int>s_subsequences_map;
 for(auto &i : s_subsequences) s_subsequences_map[i]++;
  
 //Counting the number of distinct subsequnces
 int count = 0;
 for(auto &i : s_subsequences_map){
  if(i.first == t){
   count += i.second;
  }
 }
 return count;
}

};
```

&nbsp;
**⌛ Time Complexity**  : O(2^n) and other iterative loops.
**🚀 Space Complexity** : O(n) (recursive stack)
&nbsp;

🔴 NOTE: This approach will work for shorter strings. As soon as the string size increases, time complexity will also shoot up.

-----------------------------------------------------------------------------------------------------------

**✅ *Approach 2 : Using pow for creating the subsequences***

The count of subsequences for a given string will increase beyond the capability of bit manipulation. Will give realtime error&nbsp;

```cpp
class Solution {
public:
vector<string> generateallsubsequnces(string s){
 vector<string>res;
 int size_string = s.size();
 unsigned long long int subsequence_count = pow(2,size_string);

 for(unsigned long long int  i = 0; i < subsequence_count; i++){
  string temp_string = "";
  for(unsigned long long int j = 0; j < size_string; j++){
   if((i & (1<<j))){
    temp_string.push_back(s[j]);
   }
  }
  res.push_back(temp_string);
 }
 return res;
}


int numDistinct(string s, string t) {
 vector<string>s_subsequences = generateallsubsequnces(s);
 
 //mapping unique subsequnces of s
 unordered_map<string,int>s_subsequences_map;
 for(auto &i : s_subsequences) s_subsequences_map[i]++;
  
 //Counting the number of distinct subsequnces
 int count = 0;
 for(auto &i : s_subsequences_map){
  if(i.first == t){
   count += i.second;
  }
 }
 return count;
}
};
```

&nbsp;
**⌛ Time Complexity**  : O(2^n) and other iterative loops.
**🚀 Space Complexity** : O(n)
&nbsp;

🔴 This approach will work for shorter strings. As soon as the string size increases, time complexity will also shoot up and their will be a runtime error:

`**runtime error: shift exponent 32 is too large for 32-bit type 'int' (solution.cpp)
SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior prog_joined.cpp:20:14**`

-----------------------------------------------------------------------------------------------------------

**✅ *Approach 3 : Using DP (***

We have to find distinct subsequences of S2 in S1. As there is no uniformity in data, there is no other way to find out than to **try out all possible ways**. To do so we will need to use **recursion**.

***🤖 Algorithm***

**Step 1:** Express the problem in terms of indexes.

We are given two strings. We can represent them with the help of two indexes i and j. Initially, i=n-1 and j=m-1, where n and m are lengths of strings S1 and S2. Initially, we will call f(n-1,j-1), which means the count of all subsequences of string S2[0…m-1] in string S1[0…n-1]. We can generalize it as follows:

![rec1](/assets_for_posts/dsa/dp/rec1.jpg)

**Step 2:** Try out all possible choices at a given index.

Now, i and j represent two characters from strings S1 and S2 respectively. We want to find distinct subsequences. There are only two options that make sense: either the characters represented by i and j match or they don’t.

***Case 1: When the characters match***

**if(S1[i]==S2[j])**, let’s understand it with the following example:.&nbsp;

![rec2](/assets_for_posts/dsa/dp/rec2.jpg)

S1[i] == S2[j], now as the characters at i and j match, we would want to check the possibility of the remaining characters of S2 in S1 therefore we reduce the length of both the strings by 1 and call the function recursively.

![rec3](/assets_for_posts/dsa/dp/rec3.jpg)

Now, if we only make the above single recursive call, we are rejecting the opportunities to find more than one subsequences because it can happen that the jth character may match with more characters in S1[0…i-1], for example where there are more occurrences of ‘g’ in S1 from which also an answer needs to be explored.

![rec4](/assets_for_posts/dsa/dp/rec4.jpg)

To explore all such possibilities, we make another recursive call in which we reduce the length of the S1 string by 1 but keep the S2 string the same, i.e we call f(i-1,j).

![rec5](/assets_for_posts/dsa/dp/rec5.jpg)

***Case 2: When the characters don’t match***

if(S1[i] != S2[j]), it means that we don’t have any other choice than to try the next character of S1 and match it with the current character S2.

![rec6](/assets_for_posts/dsa/dp/rec6.jpg)

This can be summarized as :

if(S1[i]==S2[j]), call f(i-1,j-1) and f(i-1,j).
if(S1[i]!=S2[j]), call f(i-1,j).

**Step 3:  Return the sum of choices**
As we have to return the total count, we will return the sum of f(i-1,j-1) and f(i-1,j) in case 1 and simply return f(i-1,j) in case 2.

**Base Cases:**

We are reducing i and j in our recursive relation, there can be two possibilities, either i becomes -1 or j becomes -1.

- if j<0, it means we have matched all characters of S2 with characters of S1, so we return 1.
- if i<0, it means we have checked all characters of S1 but we are not able to match all characters of S2, therefore we return 0.
The final pseudocode after steps 1, 2, and 3:

![rec7](/assets_for_posts/dsa/dp/rec7.jpg)

🔴 Steps to memoize a recursive solution:

If we draw the recursion tree, we will see that there are overlapping subproblems. In order to convert a recursive solution the following steps will be taken:

1.Create a dp array of size [n][m]. The size of S1 and S2 are n and m respectively, so the variable i will always lie between ‘0’ and ‘n-1’ and the variable j between ‘0’ and ‘m-1’.
2.We initialize the dp array to -1.
3.Whenever we want to find the answer to particular parameters (say f(i,j)), we first check whether the answer is already calculated using the dp array(i.e dp[i][j]!= -1 ). If yes, simply return the value from the dp array.
4.If not, then we are finding the answer for the given value for the first time, we will use the recursive relation as usual but before returning from the function, we will set dp[i][j] to the solution we get.

```cpp
class Solution {
public:
    int util(int i, int j, string& s, string& t, vector<vector<int>>& dp){
        if(j<0){
            return 1;
        }
        if(i<0){
            return 0;
        }
        if(dp[i][j]!=-1){
            return dp[i][j];
        }
        if(s[i]==t[j]){
            return dp[i][j] = util(i-1,j-1,s,t,dp)+util(i-1,j,s,t,dp);
        }
        return dp[i][j] = util(i-1,j,s,t,dp);
    }
    int numDistinct(string s, string t) {
        int m = s.size();
        int n = t.size();
        vector<vector<int>> dp(m,vector<int>(n,-1));
        
        return util(m-1,n-1,s,t,dp);
    }
};
```

&nbsp;
**⌛ Time Complexity**  : O(n*m) There are N*M states therefore at max ‘N*M’ new problems will be solved.
**🚀 Space Complexity** : O(n*m) + O(n+m) Reason: We are using a recursion stack space(O(n+m)) and a 2D array ( O(n*m)).

-----------------------------------------------------------------------------------------------------------
💻🐼💻 If there are any suggestions / questions / mistakes in my post, please do let me know by using the following email template: 👇

Email Id : vipulmaheshwari.deogarh@gmail.com.com
Subject : Question / Your Name
Body : Feedback / Suggestion / Any other comments / chit-chat
