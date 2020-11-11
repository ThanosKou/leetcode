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