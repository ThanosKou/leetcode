# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:

        """
        Do not return anything, modify head in-place instead.
        """
        
        # Divide list in two halves. Find half using fast and slow
        # Reverse second half
        # Merge the two halves
        
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        
        # the second list starts at slow.next
        
        sec_list = slow.next 
        slow.next = None # if we want to disconnect the two lists
        # now we will reverse sec_list
        curr = sec_list
        prev = None
        
        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt 
        sec_list = prev
        
        # now, if the list was 1->2->3  ->4->5, it will be  1->2->3  ->5->4
        # the two lists heads are: head and sec_list
        first_list = head
        
        while first_list and sec_list:
            first_list_nxt = first_list.next
            first_list.next = sec_list
            tmp = sec_list.next
            sec_list.next = first_list_nxt
            sec_list = tmp
            first_list = first_list_nxt
            
        











# https://leetcode.com/problems/reorder-list/
