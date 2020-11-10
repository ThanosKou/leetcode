# Definition for singly-linked list.
#class ListNode:
#    def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
        
class Solution:
    def removeZeroSumSublists(self, head):
        node = ListNode(0)
        node.next = head
        head = node
        current_node = head
        d,dvalues,csum = {},[],0
        
        while current_node:
            csum += current_node.val
            if csum in d:
                d[csum].next = current_node.next
                while dvalues[-1]!= csum: # this is the first time csum was observed, havent saved csum yet
                    cur_last = dvalues.pop()
                    del d[cur_last]
            else:
                d[csum] = current_node
                dvalues.append(csum)
            
            current_node = current_node.next
        return(head.next)
            