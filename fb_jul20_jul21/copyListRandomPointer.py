"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        
        if not head:
            return(head)
        nodeCopy = Node(head.val, None, None)
        d = {}
        d[head] = nodeCopy
        stack = [head]
        
        while stack:
            curr = stack.pop()
            if curr.next:
                if curr.next not in d:
                    nextCopy = Node(curr.next.val, None, None)
                    d[curr.next] = nextCopy
                    d[curr].next = nextCopy 
                    stack.append(curr.next)
                else:
                    d[curr].next = d[curr.next]
                    
            if curr.random:
                if curr.random not in d:
                    randomCopy = Node(curr.random.val, None, None)
                    d[curr.random] = randomCopy
                    d[curr].random = randomCopy 
                    stack.append(curr.random)
                else:
                    d[curr].random = d[curr.random]
        return(nodeCopy)


# https://leetcode.com/problems/copy-list-with-random-pointer/
