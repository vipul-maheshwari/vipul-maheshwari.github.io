---
layout: post
title: Reverse a Linked List 
date: '2022-07-08'
categories: lleasy
published: true
permalink: /:categories/:title 
---

### [Question Link](https://leetcode.com/problems/reverse-linked-list/)

-----------------------------------------------------------------------------------------------------------

&nbsp;
**✅ *Approach 1 : Using iterative version***

Given pointer to the head node of a linked list, the task is to reverse the linked list. We need to reverse the list by changing the links between nodes.

```cpp
Initialize three pointers prev as NULL, curr as head and next as NULL.
Iterate through the linked list. In loop, do following.

// Before changing next of current,
// store next node
next = curr->next

// Now change next of current
// This is where actual reversing happens
curr->next = prev

// Move prev and curr one step forward
prev = curr
curr = next

```cpp

&nbsp;

```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* current = head;
        ListNode *prev = NULL, *next = NULL;
 
        while (current != NULL) {
            // Store next
            next = current->next;
            // Reverse current node's pointer
            current->next = prev;
            // Move pointers one position ahead.
            prev = current;
            current = next;
        }
        head = prev; 
        return head;
    }
};
```

&nbsp;
**⌛ Time Complexity**  : O(n)
**🚀 Space Complexity** : O(1)

-----------------------------------------------------------------------------------------------------------

&nbsp;
**✅ *Approach 2 : Using Recursion***

1) Divide the list in two parts - first node and
2) Call reverse for the rest of the linked list.
3) Link rest to first.
4) Fix head pointer

```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
       if (head == NULL || head->next == NULL)
            return head;
 
        /* reverse the rest list and put
          the first element at the end */
        ListNode* rest = reverseList(head->next);
        head->next->next = head;
 
        /* tricky step -- see the diagram */
        head->next = NULL;
 
        /* fix the head pointer */
        return rest;
    }
};
```

&nbsp;
**⌛ Time Complexity**  : O(n)
**🚀 Space Complexity** : O(n)
&nbsp;

-----------------------------------------------------------------------------------------------------------
💻🐼💻 If there are any suggestions / questions / mistakes in my post, please do let me know by using the following email template: 👇

Email Id : vipulmaheshwari09042001@gmail.com
Subject : Question / Your Name
Body : Feedback / Suggestion / Any other comments / chit-chat
