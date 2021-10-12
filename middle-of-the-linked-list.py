class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        
        lenList = 0
        cur = head
        while cur:
            lenList += 1
            cur = cur.next
        
        #if lenList == 1:
        #    return(head.val)
        for i in range(lenList//2):
            head = head.next
        return(head)
    
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        
        fast = slow = head
        
        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next 
        
        if fast.next:
            return(slow.next)
        return(slow)
    
    # https://leetcode.com/problems/reorder-list/
