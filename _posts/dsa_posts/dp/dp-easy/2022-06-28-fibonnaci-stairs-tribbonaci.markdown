---
layout: post
title: Fibonacci Stairs(Tribbonaci)
date: '2022-06-28'
categories: dpeasy
published: true
permalink: /:categories/:title
---

### [Question Link](https://leetcode.com/problems/fibonacci-number/)

-----------------------------------------------------------------------------------------------------------

&nbsp;
**✅ *Approach 1 : Using Top-Down (Recursion)***
For the given n, we can find the nth Fibonacci number using the following recursive formula: F(n) = F(n-1) + F(n-2).

```cpp
class solution{
public:
    int fib(int n){
        if(n <= 1) return n;
        return fib(n-1) + fib(n-2);
    }
};
```

&nbsp;
**⌛ Time Complexity** : O(2^n) - exponential [time complexity](https://www.youtube.com/watch?v=pqivnzmSbq4&ab_channel=mycodeschool)
**🚀 Space Complexity** : O(n) where n shows the recursive call stack.

&nbsp;
***🧬 When we are standalone using recursive algorithm, we need to recalulate different sub values which were calculated earlier, and cost us more time to reach to the solution, as using a single dimension dp will make sure we don't need to recalculate the sub values again.. Just for the instance, if we use the recursive algorithm to find the fibonacci number for n = 100, it will take years***

![Fibonacci](/assets_for_posts/dsa/dp/Fibonnaci/recursion.jpg)
&nbsp;

-----------------------------------------------------------------------------------------------------------

&nbsp;
**✅ *Approach 2 : Using Single 1-D DP.***

What if we can use a single 1-D array to store the values of F(n-1) and F(n-2) and use the same recursive formula to calculate the F(n). That is instead of recalculating the values of F(n-1) and F(n-2) again and again, we can use the values of F(n-1) and F(n-2) to calculate F(n).

```cpp
int fibonacci(int n, vector<int>&dp){
    
    //base case
    if(n <= 1) return n;
    
    //checking if the value for the given number is already stored in the vector
    if(dp[n] != -1) return dp[n];

    //if not, store the result in the vector and make a recursive call to compute the lower members of the fibonacci series
    else{
        dp[n] = fibonacci(n-1,dp) + fibonacci(n-2,dp);
    }
    return dp[n];
}

int fib(int n) {

    //creating a vector to store the fibonacci series
    vector<int>dp(n+1,-1);
    return fibonacci(n,dp);
}
```

&nbsp;
**⌛ Time Complexity** : O(n) - linear time complexity, as we are using a single 1-D array to store the values of F(n-1) and F(n-2)
**🚀 Space Complexity** : O(n) - using a single 1-D array to store the values of F(n-1) and F(n-2)

&nbsp;
***🧬 Isn't the Time complexity should be exponential again as we are still using the two recursive calls  to find thesolution!!!! No, it's not as in case of Dynamic programming, Function calling will be same but the values are stored once a distinct function is executed and the values will be retrieved if the same function calls again.***
&nbsp;

-----------------------------------------------------------------------------------------------------------

&nbsp;
**✅ *Approach 3 : Using Memorisation : Bottom up approach***

As we already know, for calculating nth Fibonacci number, we only need it's two previous counterparts. For example if we want to calculate the 3rd Fibonacci number, we only need the 2nd Fibonacci number and the 1st Fibonacci number. Utilizing the fact, we can use 2 variables to store the two previous values and they will keep changing according to the new states.. Using this approach, we can solve the problem using O(n) time complexity and O(1) space complexity. Best of both hands..

```cpp
class Solution {
public:
    int fib(int n) {
    int prev2 = 0, prev1 = 1, curr, i;
    if( n == 0)
        return prev2;
    for(i = 2; i <= n; i++)
    {
       curr = prev2 + prev1;
       prev2 = prev1;
       prev1 = curr;
    }
    return prev1; 
    }
};
```

&nbsp;
**⌛ Time Complexity** : O(n) - linear time complexity, as we are using two variables to store the two previous values
**🚀 Space Complexity** : O(1) - using two variables to store the two previous values
&nbsp;

-----------------------------------------------------------------------------------------------------------

&nbsp;
⭐ [Climbing stairs](https://leetcode.com/problems/climbing-stairs/) is the exactly same problem as Fibonacci numbers. As we know, the Fibonacci number is the sum of two previous Fibonacci numbers. Similarly, To climb n stairs, we need to find the different approaches to climb the n-1 and n-2 stairs. Looks fimiliar isn't it!!!

⭐ [N-th Tribonacci Number](https://leetcode.com/problems/n-th-tribonacci-number/) instead of calculating two previous sub values, here we need three of them. That is F(n) = F(n-1) + F(n-2) + F(n-3). And bingo! It's done...
&nbsp;

-----------------------------------------------------------------------------------------------------------
💻🐼💻 If there are any suggestions / questions / mistakes in my post, please do let me know by using the following email template: 👇

Email Id : vipulmaheshwari.deogarh@gmail.com
Subject : Question / Your Name
Body : Feedback / Suggestion / Any other comments / chit-chat
