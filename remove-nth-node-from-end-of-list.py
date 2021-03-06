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