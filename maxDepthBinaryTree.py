# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        
        res = 0
        if not root:
            return(0)
        
        stack = [root]
        d = {}
        d[root] = 1
        while stack:
            curr = stack.pop()
            if curr.left:
                stack.append(curr.left)
                d[curr.left] = d[curr] + 1

            if curr.right:
                stack.append(curr.right)
                d[curr.right] = d[curr] + 1
        
        return(max(d.values()))
      
      # recursive 
      
      # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return(0)
        return(1 + max(self.maxDepth(root.left), self.maxDepth(root.right)))
      
  # https://leetcode.com/problems/maximum-depth-of-binary-tree/
