---
layout: post
title: Remove Nth Node From the End of the LL
date: '2022-07-17'
categories: lleasy
published: true
permalink: /:categories/:title 
---

### [Remove Nth Node From the End of the LL](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)

-----------------------------------------------------------------------------------------------------------
&nbsp;
**✅ *Approach 1 : Brute force ( Two passes)***

Doing first pass can easily find the total number of nodes in the linked list. Second pass will let you remove the nth node from the end or N-nth node from the start. naïve,NOT appropriate..&nbsp;

-----------------------------------------------------------------------------------------------------------
&nbsp;
**✅ *Approach 1 : Use Fast and Slow pointers***

Using two iterations for finding the nth node from the end of the linked list is so naïve. Instead we can maintain a gap of n number of nodes and using the concept of fast and slow pointers, we can reach to the nth node from the end of the linked list..&nbsp;

```cpp
class Solution {
public:
ListNode* removeNthFromEnd(ListNode* head, int n) {
 ListNode *fast = head, *slow = head;
 while(n--) fast = fast -> next;      // iterate first n nodes using fast
 if(!fast) return head -> next;       // if fast is already null, it means we have to delete head itself. So, just return next of head
 while(fast -> next)                  // iterate till fast reaches the last node of list
  fast = fast -> next, slow = slow -> next;            
 slow -> next = slow -> next -> next; // remove the nth node from last
 return head;
}
};
```

&nbsp;
**⌛ Time Complexity**  : O(n) where N is the number of nodes in the given list. Although, the time complexity is same as two iteration but, we have reduced the constant factor in it to half.
**🚀 Space Complexity** : O(1) since only constant space is used.

-----------------------------------------------------------------------------------------------------------
💻🐼💻 If there are any suggestions / questions / mistakes in my post, please do let me know by using the following email template: 👇

Email Id : vipulmaheshwari.deogarh@gmail.com
Subject : Question / Your Name
Body : Feedback / Suggestion / Any other comments / chit-chat
