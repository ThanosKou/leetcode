# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        
        preorder = collections.deque(preorder)
        return(self.recurse(preorder, inorder))
    
    def recurse(self, preorder, inorder):
        if inorder:
            node = preorder.popleft()
            i =  inorder.index(node)
            root = TreeNode(node)
            root.left = self.recurse(preorder, inorder[:i])
            root.right = self.recurse(preorder, inorder[i+1:])

            return(root)

            

# https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
