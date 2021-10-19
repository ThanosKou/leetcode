# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return(root)
        level = [root]
        res = []
        while level:
            level_res = []
            queue = []
            for node in level:
                level_res.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level_res)
            level = queue
        return(res)

# https://leetcode.com/problems/binary-tree-level-order-traversal/
