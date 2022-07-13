---
layout: post
title: Best time to buy and sell stock
date: '2022-06-09'
categories: arrayeasy
published: true
permalink: /:categories/:title
---

### [Question Link : Best time to buy and sell stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

-----------------------------------------------------------------------------------------------------------

&nbsp;
**✅ *Approach 1 : Using Brute force (Single iteration)***

- We can easily solve this problem using brute force.
- Just iterate over the whole array and if the current price of the stock is less than that of the previous buying price, we buy the stock.
- If the current price of the stock is greater than that of the previous buying price, we sell the stock, if the profit for that day is greater than the current profit, we update the profit.

```cpp
class Solution {
public:
int maxProfit(vector<int>& prices) {
    int buying_price = INT_MAX, curr_profit = 0, profit = 0;
    for(int i = 0; i < prices.size(); i++) {
        buying_price = min(buying_price, prices[i]);
        profit = max(profit,prices[i] - buying_price);
    }
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
