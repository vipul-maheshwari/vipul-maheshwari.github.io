---
layout: post
title: Roman to Integer
date: '2022-06-10'
categories: arrayeasy
published: true
permalink: /:categories/:title 
---

### [Question Link : Roman to integer](https://leetcode.com/problems/roman-to-integer/)

-----------------------------------------------------------------------------------------------------------

&nbsp;
**✅ *Approach 1 : Using Unordered MASKING***

As the roman numerals are fixed, we can use an unordered map to store the value of each literal and using it to calculate the overall value.
&nbsp;

```cpp
class Solution {
public:
int romanToInt(string s) {

    unordered_map<char, int> values;
    values['I'] = 1; values['V'] = 5; values['X'] = 10; values['L'] = 50; values['C'] = 100; values['D'] = 500; values['M'] = 1000;

    int sum = 0, i = 0;
    while(i < s.size()){
        
        //For those edging conditions : IV, IX, IL, IC, ID, IM, VL
        if(i < s.size() - 1 && values[s[i]] < values[s[i + 1]]){
            sum += values[s[i + 1]] - values[s[i]];
            i += 2;
        }
        else{
            sum += values[s[i]];
            i++;
        }
    }

}
};
```

&nbsp;
**⌛ Time Complexity**  : O(n) where n is the length of the string
**🚀 Space Complexity** : O(m) where m is the number of Roman literals

-----------------------------------------------------------------------------------------------------------
💻🐼💻 If there are any suggestions / questions / mistakes in my post, please do let me know by using the following email template: 👇

Email Id : vipulmaheshwari09042001@gmail.com
Subject : Question / Your Name
Body : Feedback / Suggestion / Any other comments / chit-chat
