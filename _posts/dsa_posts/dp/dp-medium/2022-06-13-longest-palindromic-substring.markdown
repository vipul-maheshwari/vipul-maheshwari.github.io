---
layout: post
title: Longest Palindromic Substring
date: '2022-06-13'
categories: dpmedium
published: true
permalink: /:categories/:title
---

### [Question Link : Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/) -- [Solution](https://youtu.be/XYQecbcd6_c)

-----------------------------------------------------------------------------------------------------------

&nbsp;
**✅ *Approach 1 : Using Brute force***

Solving this problem with the brute force approach is easy. We can iterate over every possible substring of the given string and check if it is a palindrome or not... But it would take O(n^3) time. As we need two for loops to find the possible substrings, and one another for checking if it is a palindrome, or not.

So, overall this is a bad idea if the number of characters in the string is very large.

```cpp
class Solution {
public:
bool returnpalindrome(string str){
    int left = 0, right = str.size() - 1;
    while(left < right){
        if(str[left] != str[right]){
            return false;
        }
    }
    return true;
}

vector<string> returnsubstrings(string s){
    vector<string>substrings;
    for(int i = 0; i < s.size(); i++){
        for(int j = i; j < s.size(); j++){
            substrings.push_back(s.substr(i,j-i+1));
        }
    }
}

string longestPalindrome(string s) {
    vector<string>substrings_s = returnsubstrings(s);
    string longest = "";
    for(int i = 0; i < substrings_s.size(); i++){
        if(returnpalindrome(substrings_s[i]) && substrings_s[i].size() > longest.size()){
            longest = substrings_s[i];
        }
    }
    return longest;
}
};
```

&nbsp;
**⌛ Time Complexity** : O(n^3)
**🚀 Space Complexity** : O(n)

-----------------------------------------------------------------------------------------------------------

**✅ *Approach 2 : Using 'Expand around centre' technique***

We know that a palindrome mirrors itself around its centre. As a result, a palindrome can be expanded from the centre. So what if we fix a centre and expand in both directions for longer palindromes? So we can observe that: The objective is to consider one by one every index as a centre and generate all even and odd length palindromes while keeping track of the longest palindrome seen so far.

For odd length palindromes:

Fix a centre and expand in both directions, i.e., fix i (index) as the centre and two indices as i1 = i+1 and i2 = i-1. If i1 and i2 are equal, then decrease i2 and increase i1 to find the maximum length.

For even length palindromes:

Take two indices, i1 = i and i2 = i+1. Compare characters at i1 and i2 to get the maximum length until all pairs of compared characters are equal. i.e while str[i1] == str[i2].

```cpp
class Solution {
public:
    string longestPalindrome(string str) {
    int n = str.length();
    if(n <= 1) return str;
    int l = 0;
    int h = 0;
    int start = 0;
    int maxlen = INT_MIN;

    for(int i = 0; i < n; i++){
        //palindrome of even length with center in space between i and i+1
        l = i;
        h = i+1;
        while(l >= 0 && h < n && str[l] == str[h]){
            if(h-l+1 > maxlen){
                start = l;
                maxlen = h-l+1;
            }
            l--;
            h++;
        }
    }

    //palindrome of odd length with center at i
    for(int i = 0; i < n-1; i++){
        l = i;
        h = i;
        while(l >= 0 && h < n && str[l] == str[h]){
            if(h-l+1 > maxlen){
                start = l;
                maxlen = h-l+1;
            }
            l--;
            h++;
        }
    }

    return str.substr(start, maxlen);
    }
};
```

&nbsp;
**⌛ Time Complexity**  : O(n^2)
**🚀 Space Complexity** : O(1)

⭐ Since the expanding of a palindrome around its centre can take O(n) time and there would be at most    (2 * n + 1) centres.

- N Characters
- N - 1 Gaps b/w Characters
- 2 Possible centres at the front and the back
- N + N - 1 + 2 = 2*N + 1

&nbsp;
For an example if the string is "abababa", there are 2 * 7 + 1 = 15 centre positions available.

***🔴 NOTE: If the even length palindrome centre span in between i and i-1 then the first character won't become the centre of the palindrome in the even length case.***

-----------------------------------------------------------------------------------------------------------
**✅ *Approach 3 : Using Dynamic programming***

Time complexity is improved from O(n^3) to O(n^2) using the center expansion technique. But we can also use DP wiht a memorisation table  to find the results by storing the occurence of the longest palindromeat each index position. Before going into the details of the algorithm, let us understand the very observations we made during the center expansion technique. Here

**🎯*Observation 1: Every individual character is palindrome. We don't need to worry about the singleone characters***
-> There is no need to check if the character is a palindrome or not every time for single characters.

**🎯*Observation 2: We don't need a function call for two characters one after another to check if they are palindrome***
-> For strings of length 2, we don't need a loop to check if they are palindrome or not. Just check if both characters are equal or not.

**🎯*Observation 3: For strings with greater size***
->Let's say for an example if the string is "bab" that is if the size is 3,  we only need to check if the str[0] and str[2] are equal or not, if they are then whole string is palindrome otherwise not. For strings greater than 3 we need the information regarding the strings of lesser sizes. Let's say if the string is "cdbabde" and now we want to check if the whole string is palindrome or not, that is from index 0 to index 6, for that, instead of checking the whole string we will only look for the substring "dbabd" starting from the index 1 and ending at 5 and whose length is 1 less than that of the original string. (NOTE: When we are working with DP problems, we already know the results of sub problems and we can use them to solve the current problem.) As the substring "dbabd" is palindrome, now just check if the 0th index character is equal to the 6th index character. If they are equal then the whole string is palindrome otherwise not. As character c is not equal to character e, so the string is not palindrome.

We can store these results in an array. This suggests the use of dynamic programming.

From the information above, we need a memorisation table. Therefore we use a 2D boolean array where the rows represent i, and columns represent j. Let's call this array dp[][], and dp[i][j] will store whether the substring from ith position to jth is a palindrome or not.

As now we are using the table to find if the string is palindrome or not, we don't need any kind of boolean function to check if the string is palindrome or not. So we are just removing the function and replacing it with a logic to fill the table.

Case 1:

i==j
Every single character of a string is a palindrome. Therefore dp[i][j] is true. (Fill all the diagonals of the table with true)

Case 2:

j-i=1
We're checking two characters at a time, if s[i]==s[j], then dp[i][j] is true else false. (Fill the diagonals of the table with true if the characters are equal)

Case 3:

j-i>=2
Consider a string as "abcb" and we are checking palindromic condition for the substring starting at 1th index and ending at 3rd index that is "bcb". Now using the memorisation table, we are currently standing at 1th row and 3rd column (see the figure below). As both 1st and 3rd characters are equal we will take up the result at dp[i+1][j-1] as the i+1 and j-1 coordinates are eliminating both b's from the substring and as the character c is singleone character the dp[i+1][j-1] is a diagonal position, it is always true, and hence we will fill true for the dp[i][j] for this case. This whole process is repeated for all the cases.&nbsp;

![table](/assets_for_posts/dsa/dp/Longest-palindrome-substring/table.jpg)

```cpp
class Solution {
public:
    string longestPalindrome(string str) {
    int n = str.length();
    if(n <= 1) return str;
    int x = 0;
    int y = 0;
    int start = 0;
    int maxlen = 1;
    vector<vector<bool>>dp(n, vector<bool>(n, false));
    for(int i = n-1; i >= 0; i--){
        for(int j = n-1; j >= i; j--){
            if(i == j){
                dp[i][j] = true;
            }
            else if(j-i == 1){
                dp[i][j] = (str[i] == str[j]);
            }
            else{
                dp[i][j] = (str[i] == str[j]) && dp[i+1][j-1];
            }

            if(dp[i][j] && j-i+1 > maxlen){
                maxlen = j-i+1;
                start = i;
            }
    }
    }
    return str.substr(start, maxlen);
    }
};
```

&nbsp;
**⌛ Time Complexity**  : O(n^2)
**🚀 Space Complexity** : O(n^2)

⭐ The approach described here can be used to solve various dynamic programming questions such as LCS.

⭐ The dynamic programming approach is very useful when it comes to optimization problems where we can solve sub problems and combine their final results to reach the last final solution such as graph algorithms (All pairs shortest path algorithm).

-----------------------------------------------------------------------------------------------------------
**✅ *Approach 3 : Using Manacher's Algorithm***

Manacher's algorithm combines the ideas which are explored in the above two approaches.

- DP : Storing the palindrome lengths after every index position.
- Central Expansion : For efficient iteration.

Understanding the Manacher's algorithm in the first time was a bit tricky. I recommend this [YouTube Video](https://youtu.be/nbTSfrEfo6M) and [Article](https://www.geeksforgeeks.org/manachers-algorithm-linear-time-longest-palindromic-substring-part-1/) for better understanding of the algorithm. The algorithm is simple and it is also efficient.

It is using the fact that the left and right sides of a palindrome are very mirror images of each other, as it avoids re-checking of the palindromes that have already been found earlier. In the layman terms, if there is a shorter palindrome which is located to the left of the center of the longer palindrome, then this short palindrome is confirmed to be present on the right side of the longer palindrome.

***🔴 So when you reach to the centre of the shorter palindrome on the right side, the expansion can begin at the earlier known ends of the palindrome rather than from the center again.***

Here are the steps to understand the algorithm:

***🤖 Algorithm***
Step 1: Insert another 2 different special characters in the front and the end of the given string to avoid bound checking.
Step 2: Convert the string to the centre format, where every original character is surrounded by a '#' character representing the gap b/w two characters.
Step 3: Start scaning the string from i = 1 to i = n-1. For each i, we will find it's mirror character's position in the string.
Step 4: If the current index is less than the right boundary then we have two cases:
    - Case 1: Take the mirror position's palindromic length
    - Case 2: Take the current position's palindromic length which is equal to the right boundary minus the current position.
    Then we will find the minimum of two values...
Step 5: After updating the current position's palindromic length, we will exapnd considering that position as the centre of the palindrome.
Step 6: If the exapanded palindrome is greater than the current palindrome length, then we will update the current palindrome length as well as the right boundary and the current centre of the palindrome.
Step 7: Return the substring starting from the left boundary of the current palindrome and ending at the right boundary.

***🔴 NOTE: You might need to go through the video and article as I have mentioned above to grasp the proper logic and proper understanding behind this algorithm.***

```cpp
class Solution {
public:
    string longestPalindrome(string s) {
        // Insert another 2 different special characters in the front and the end of string s to avoid bound checking.
        string manacher_str = "@#";
        for (int i = 0; i < s.size(); i++)
            manacher_str = manacher_str + s[i] + "#";
        manacher_str += "$";
        
        int len = manacher_str.size();
        vector<int> RL(len, 0);
        
        int c = 0, r = 0;    // current center, right limit
        int maxLen = 0, maxLenPos = 0;  //maxLen
        
        for (int i = 1; i < len - 1; i++) {
            
            // find the corresponding mirror letter in the palidrome subString
            int iMirror = 2 * c - i;
            
            //If the current index is less than the right boundary:
            //Case 1: Take the mirror position palindromic length
            //Case 2: Take the current position palindromic length which is the Right boundary's position minus the current position
            if (i < r)
                RL[i] = min(RL[iMirror], r - i);

            //Now we are going to expand the palindrome around the current position till we are getting a palindromic substring
            //The expansion should take place within the limits of the last and the first index of the given string
            //Increment the count of the length of the palindrome for the given centre position.
            while (i - RL[i] >= 0 && i + RL[i] < len && manacher_str[i + RL[i] + 1] == manacher_str[i - RL[i] - 1])
                RL[i]++;
            
            //If the palindromic length for the given centre is greater than that of the previous maximum length, update the results.
            if (RL[i] > maxLen) {
                maxLen = RL[i];
                maxLenPos = i;
            }
            
            
            // Update c, r in case if the palindrome centered at i expands beyond the right boundary of the previous palindrome.
            if (i + RL[i] > r) {
                r = i + RL[i];
                c = i;
            }
        }
        
        return s.substr((maxLenPos - 1 - maxLen) / 2, maxLen);
    }
};
```

&nbsp;
**⌛ Time Complexity**  : O(n)
**🚀 Space Complexity** : O(n)

⭐ Iterating over 2*N+1 characters is the reason why the algorithm is O(n) time complexity.

⭐ There are 2*N+3 characters (Considering the extra characters inserted at the front and end of the string) in the given string. Making Space Complexity as O(n)

⭐ If you didn't get the Manacher's algorithm in the first time, don't beat yourself, I spent more than 4 hours to figure out the algorithm.

-----------------------------------------------------------------------------------------------------------
💻🐼💻 If there are any suggestions / questions / mistakes in my post, please do let me know by using the following email template: 👇

Email Id : vipulmaheshwari.deogarh@gmail.com
Subject : Question / Your Name
Body : Feedback / Suggestion / Any other comments / chit-chat
