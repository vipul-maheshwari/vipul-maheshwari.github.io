---
layout: post
title: Merge Two Sorted Linked List
date: '2022-07-16'
categories: lleasy
published: true
permalink: /:categories/:title 
---

### [Merge Two Sorted Linked List](https://leetcode.com/problems/merge-two-sorted-lists/)

-----------------------------------------------------------------------------------------------------------
&nbsp;
**✅ *Approach 1 : Using an External Linked List to store the the answers.***

***🤖 Algorithm***
Step 1: Create a new dummy node. It will have the value 0 and will point to NULL respectively. This will be the head of the new list. Another pointer to keep track of traversals in the new list.

Step 2:  Find the smallest among two nodes pointed by the head pointer of both input lists, and store that data in a new list created.

Step 3: Move the head pointer to the next node of the list whose value is stored in the new list.

Step 4: Repeat the above steps till any one of the head pointers stores NULL. Copy remaining nodes of the list whose head is not NULL in the new list.&nbsp;

🔴Note: The above algorithm is not the most efficient way to merge two sorted linked lists. Instead we can use the merge sort algorithm to merge two sorted linked lists.

-----------------------------------------------------------------------------------------------------------
&nbsp;
**✅ *Approach 1 : Use Fast and Slow pointers***

Use a dummy pointer to iterate through both of the sorted linked list, change the next pointer of dummy pointer accordingly.&nbsp;

***🤖 Algorithm***
Step 1: Create two pointers, say l1 and l2. Compare the first node of both lists and find the small among the two. Assign pointer l1 to the smaller value node.

Step 2: Create a pointer, say res, to l1. An iteration is basically iterating through both lists till the value pointed by l1 is less than or equal to the value pointed by l2.

Step 3: Start iteration. Create a variable, say, temp. It will keep track of the last node sorted list in an iteration.

Step 4: Once an iteration is complete, link node pointed by temp to node pointed by l2. Swap l1 and l2.

Step 5: If any one of the pointers among l1 and l2 is NULL, then move the node pointed by temp to the next higher value node.

```cpp
ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
 
 ListNode *dummy = new ListNode();
 ListNode *temp = dummy;

 //We will iterate till we don't reach to the end of the linked list of either one of them
 while(list1 != NULL && list2 != NULL){

  if(list1->val < list2->val){
   temp->next = list1;
   list1 = list1->next;
  }
  else{
   temp->next = list2;
   list2 = list2->next;
  }
  temp = temp->next;
 }

 while(list1 != NULL){
  temp->next = list1;
  list1 = list1->next;
  temp = temp->next; 
 }

 while(list2 != NULL){
  temp->next = list2;
  list2 = list2->next;
  temp = temp->next;
 }
 return dummy->next;
}
```

&nbsp;
**⌛ Time Complexity**  : O(n)
**🚀 Space Complexity** : O(1)

-----------------------------------------------------------------------------------------------------------
💻🐼💻 If there are any suggestions / questions / mistakes in my post, please do let me know by using the following email template: 👇

Email Id : vipulmaheshwari.deogarh@gmail.com
Subject : Question / Your Name
Body : Feedback / Suggestion / Any other comments / chit-chat
