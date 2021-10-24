# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        
        if not root:
            return(root)
        
        queue = collections.deque([root])
        level = queue
        res = []
        
        while queue:
            queue = collections.deque([])
            while level:
                curr = level.popleft()
                final_val = curr.val
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)
            level = queue
            res.append(final_val)

        return(res)


#  https://leetcode.com/problems/binary-tree-right-side-view/
