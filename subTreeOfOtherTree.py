# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        
        subTree = ''.join(self.dfs_print(subRoot))
        mainTree = ''.join(self.dfs_print(root))
        return(subTree in mainTree)
        
    def dfs_print(self, root):
        res = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node:
                res += '<' + str(node.val) + '>'
                stack.append(node.left)
                stack.append(node.right)
            else:
                res.append('*')
        
        return(res)
    
 # recursive

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        
        def dfs(root, subRoot):
            if not root:
                return(False)
            if root.val == subRoot.val and CheckTrees(root, subRoot):
                return(True)

            return(dfs(root.left, subRoot) or dfs(root.right, subRoot))

        def CheckTrees(tree1, tree2):

            if not tree1 and not tree2:
                return(True)

            if tree1 and not tree2 or not tree1 and tree2:
                return(False)

            if tree1.val != tree2.val:
                return(False)

            return CheckTrees(tree1.left, tree2.left) and CheckTrees(tree1.right, tree2.right)
        
        if not subRoot:
            return(True)
        
        return(dfs(root, subRoot))
            
# https://leetcode.com/problems/subtree-of-another-tree/
