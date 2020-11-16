# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    ## Recursion
    #def searchBST(self, root: TreeNode, val: int) -> TreeNode:
    #    if root.val == val:
    #        return(root)
    #    elif root.val > val and root.left:
    #        return(self.searchBST(root.left,val))
    #    elif root.val < val and root.right:
    #        return(self.searchBST(root.right,val))
    #    else: 
    #        return(None)
        
    ## Binary search
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        l = root
        while l:
            if l.val == val:
                return(l)
            elif l.val > val:
                l = l.left
            else:
                l = l.right
        return(None)
        