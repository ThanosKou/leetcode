# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        
        node_min = float('-inf')
        node_max = float('inf')   
        bfs_queue = collections.deque([(root, node_min, node_max)])
        
        # Basically, we keep local minima and maxima for every node.
        # At the beginning, root < inf, and root > -inf.
        # then, as we do bfs with queue, we not only store the node, but also its local minima and maxima
        # if left node (node.left.val), then it needs to be < its parent (node.val) and > local minima
        # if right node (node.right.val), then it needs to be > its parent (node.val) and < local maxima
        # A left node always updates its local maxima with their own parent
        # A right node always updates its local minima with their own parent
        
        while bfs_queue:
            node, node_min, node_max = bfs_queue.popleft()
            if node.left:
                if node.left.val <= node_min or node.left.val >= node.val: 
                    return(False)
                bfs_queue.append((node.left, node_min, node.val))
            if node.right:
                if node.right.val <= node.val or node.right.val >= node_max:
                    return(False)
                bfs_queue.append((node.right, node.val, node_max))
      
        return(True)
    
## recursive
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        
        output = []
        self.inOrderTrav(root, output)
        
        for i in range(1, len(output)):
            if output[i-1] >= output[i]:
                return(False)
        return(True)
    
    def inOrderTrav(self, root, output):
        if not root:
            return
        
        self.inOrderTrav(root.left, output)
        output.append(root.val)
        self.inOrderTrav(root.right, output)
      

    
# https://leetcode.com/problems/validate-binary-search-tree/
