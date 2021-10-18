# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# when we apply queue to that binary tree, we do BFS!
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        
        queue = collections.deque([(p,q)])
        while queue:
            p,q = queue.popleft()
            if not p and not q:
                continue
            if not (p and q):
                return(False)
            if p.val != q.val:
                return(False)
            queue.append((p.left, q.left))                     
            queue.append((p.right, q.right))
        
        return(True)
      
# when we apply stack, we do DFS !
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        
        stack = [(p,q)]
        while stack:
            p,q = stack.pop()
            if not p and not q:
                continue
            if not (p and q):
                return(False)
            if p.val != q.val:
                return(False)
            stack.append((p.left, q.left))                     
            stack.append((p.right, q.right))
        
        return(True)
      
      # https://leetcode.com/problems/same-tree/
