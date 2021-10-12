class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        fast = head
        slow = head
        
        while n > 0:
            fast = fast.next
            n -= 1
            
        if not fast: 
            return head.next
        
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        
        return head
    
    
    class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        
        dummy = ListNode(0)
        dummy.next= head
        fast = slow = dummy
        
        for i in range(n):
            fast = fast.next # make it n steps ahead
        
        if not fast:
            return(fast)
        
        while fast.next:
            fast = fast.next
            slow = slow.next
        
        # now slow.next points to the node that we want to delete
        
        slow.next = slow.next.next
        return(dummy.next)
    
    # https://leetcode.com/problems/remove-nth-node-from-end-of-list/
