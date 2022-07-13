---
layout: post
title: Buy and sell stock 2
date: '2022-06-09'
categories: arraymedium
published: true
permalink: /:categories/:title 
---

### [Question Link : Buy and sell stock 2 ||](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)

-----------------------------------------------------------------------------------------------------------

&nbsp;
**✅ *Approach 1 : Using Two pointers***

Unlike in [Best time to buy and sell stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) here we can do multiple transactions. That is you can buy and sell the stock on the same day or buy yesterday sell today, buy today sell next day, every possibility is open considering that you make profit out of each transaction.

The question is how to find which day to buy and which day to sell.

To make the most proift, We need to look out for the local valley and the peaks. Buy at the valley, sell at the peaks. So that we can make the profits at each possible opportunity.&nbsp;

![valley-peak](/assets_for_posts/dsa/array/easy/best-time-to-buy-and-sell-stock-2/valley-peak.jpg)

```cpp
//example:[7,1,5,3,6,4]

class Solution {
public:
int maxProfit(vector<int>& prices) {
    
    int buy, sell, profit = 0;
    int i = 1, n = prices.size();
    while(i < n){

        //Finding the local valley
        for(buy = prices[i-1]; i < n && prices[i] < buy; i++){
            buy = prices[i];
        }
        
        //If we reached to the edge of the mountain
        if(i >= n){
            break;
        }
        
        //Finding the local Peak
        for(sell = prices[i]; i < n && prices[i] >= sell; i++){
            sell = prices[i];
        }
        
        //Making profit at each mountain range
        profit += sell-buy;
        i++;
    }
    return profit;
}
};
```

&nbsp;
**⌛ Time Complexity**  : O(n)
**🚀 Space Complexity** : O(1)

-----------------------------------------------------------------------------------------------------------

**✅ *Approach 2 : Using one pass iteration (Better version of the approach 1)***

Instead of calculating local valleys and peaks, what if we start to look out for the profits at each day considering the price of the stock on the previous day.

For example, if the stock price of a given stock is [1,2,3,4,5,6] then if we solve this problem with mountain ambition, that is finding the local valleys and peaks, the maximum profit would be 5 when we buy the stock on the day 1 and sell it on the day 6.

But let’s say I buy the stock at day 1, and as soon as it’s day 2 the price of the stock is greater than it’s previous day, as there is an opportunity to make profit, hence the profit is 1 at day 2.

Now I buy the stock at day 2 and sell it on the day 3, profit increases to 2, henceforth it continues till I reach at the end and the profit becomes 5 which is same as if we would have solve the problem using the mountain technique of valleys and peaks..&nbsp;

```cpp
class Solution {
public:
    int maxProfit(vector<int>& P) {
        int profit = 0;
        for(int i = 1; i < size(P); i++)
            if(P[i] > P[i-1])              // yesterday was valley, today is peak
                profit += P[i] - P[i-1];   // buy yesterday, sell today
        return profit;
    }
};
```

&nbsp;
**⌛ Time Complexity**  : O(n)
**🚀 Space Complexity** : O(1)

-----------------------------------------------------------------------------------------------------------
💻🐼💻 If there are any suggestions / questions / mistakes in my post, please do let me know by using the following email template: 👇

Email Id : vipulmaheshwari09042001@gmail.com
Subject : Question / Name
Body : Feedback / Suggestion / Any other comments / chit-chat
