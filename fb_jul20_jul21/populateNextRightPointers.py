"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        
        if not root:
            return(root)
        queue = collections.deque([root])
        level = collections.deque([root])
        res = []
        
        while queue:
            cur_level = []
            queue = collections.deque([])
            while level:
                node = level.popleft()
                cur_level.append(node)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            level = queue
            res.append(cur_level)
        
        # res contains all tree nodes grouped by level
        
        for level in res:
            for i in range(len(level)-1):
                level[i].next = level[i+1]
            #level[-1].next = None
        
        return(root)

    ## But even better:
    
    """
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        
        if not root:
            return(root)
        
        curr = root
        nxt = root.left # mark this as start of your next level
        
        while curr.left:
            curr.left.next = curr.right
            if curr.next:    
                curr.right.next = curr.next.left
                curr = curr.next
            else:
                curr = nxt
                nxt = curr.left
                
        return(root)


# https://leetcode.com/problems/populating-next-right-pointers-in-each-node/
