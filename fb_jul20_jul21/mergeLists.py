# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        
        merged = ListNode(0)
        output = merged 
        while l1 and l2:
            if l1.val < l2.val:
                merged.next = l1
                l1 = l1.next
            else:
                merged.next = l2
                l2 = l2.next
            merged = merged.next
        merged.next = l1 or l2
        return(output.next)'
      

## In place!!

class Solution:
    def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        
        curr = dummy = ListNode(0)
        dummy.next = l1
        
        while l1 and l2:
            if l1.val <= l2.val:
                l1 = l1.next    # curr starts from l1, so we can just keep moving on
            else:               # we need to make curr pointing to l2, but dont lose l1.next !!
                l1_nxt = l1.next
                curr.next = l2    # l1 points to l2
                                 # now we want l2 to point back to l1, but don't forget l2.next !!
                tmp = l2.next
                l2.next = l1
                l2 = tmp
            curr = curr.next
        curr.next = l1 or l2 # one of them is None for sure
        return(dummy.next)
      
# https://leetcode.com/problems/merge-two-sorted-lists/
