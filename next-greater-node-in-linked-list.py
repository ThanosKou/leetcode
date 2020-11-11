# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next 
class Solution:
    def nextLargerNodes(self, head: ListNode) -> List[int]:
        
        #while cur is not None:
        #    cur = cur.next
        #    l += 1
        stack = []
        ans = []
        ind = 0
        cur = head
        while cur is not None:
            ans.append(0)
            
            while stack and cur.val > stack[-1][0]: # as soon as I find a larger guy I stop
                _,index = stack.pop()
                ans[index] = cur.val
                
            stack.append((cur.val,ind))
            ind += 1
            cur = cur.next
        return(ans)