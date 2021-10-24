# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        
        curA, curB = headA, headB
        lenA = lenB = 0
        
        while curA:
            curA = curA.next
            lenA += 1
        
        while curB:
            curB = curB.next
            lenB += 1
        
        curA, curB = headA, headB
        
        if lenA < lenB:
            for _ in range(lenB-lenA):
                curB = curB.next        
        elif lenA > lenB:
              for _ in range(lenA-lenB):
                curA = curA.next
        
        while curA != curB:
            curA = curA.next
            curB = curB.next
        
        return(curA)


# https://leetcode.com/problems/intersection-of-two-linked-lists/
