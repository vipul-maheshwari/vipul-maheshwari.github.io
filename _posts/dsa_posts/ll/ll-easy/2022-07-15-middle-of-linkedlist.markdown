---
layout: post
title: Middle of the Linked List
date: "2022-07-15"
categories: ll-easy
published: true
permalink: /:categories/:title 
---

### [Question Link : Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/)

-----------------------------------------------------------------------------------------------------------
&nbsp;
**✅ *Approach 1 : Brute force***

Find the total number of nodes in the linked list and return the middle one. naïve,NOT appropriate..&nbsp;

-----------------------------------------------------------------------------------------------------------
&nbsp;
**✅ *Approach 2 : Use Fast and Slow pointers***

Take two pointers, one slow and one fast, move fast twice as “fast” than slower one and when you reach to the end, that is when fast pointer reaches to the null or the last node, slower one will be exactly at the middle of the list..&nbsp;

```cpp
class Solution {
public:
ListNode* middleNode(ListNode* head) {
 
 //If there are less than or equal to 1 nodes in the list, return the head
 if(head == NULL || head->next == NULL) return head;

 //If there are two nodes
 if(head->next->next == NULL) return head->next;

 ListNode *slow = head, *fast = head;
 while(fast != NULL && fast->next != NULL) {
  fast = fast->next->next;
  slow = slow->next;
 }
 return slow;
}
};
```

&nbsp;
**⌛ Time Complexity**  : O(n)
**🚀 Space Complexity** : O(1)

-----------------------------------------------------------------------------------------------------------
💻🐼💻 If there are any suggestions / questions / mistakes in my post, please do let me know by using the following email template: 👇

Email Id : vipulmaheshwari.deogarh@gmail.com
Subject : Question / Your Name
Body : Feedback / Suggestion / Any other comments / chit-chat
