# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.max_sum = float('-inf')
        self.dfs(root)
        
        return(self.max_sum)
    
    def dfs(self, root):
        
        if not root:
            return(0)
        
        left_sum = max(0,self.dfs(root.left))
        right_sum = max(0,self.dfs(root.right))
        
        self.max_sum = max(self.max_sum, left_sum + right_sum + root.val) #!!! this is if we dont consider any other path
        
        # but what we return is for further tree exploration, hence only one subtree 
        return(root.val + max(left_sum,right_sum))
      
# https://leetcode.com/problems/binary-tree-maximum-path-sum/
