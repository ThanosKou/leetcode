# my solution was a bit dumb, I didnt use the fact that this is a BST, it can work with any binary tree

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        
        ancest = {root: 'start'}
        
        queue = collections.deque([root])
        while queue:
            node = queue.popleft()
            if node.left:
                ancest[node.left] = node
                queue.append(node.left)
            if node.right:
                ancest[node.right] = node
                queue.append(node.right)
        
        node, p_lca = p, [p]
        while ancest[node] != 'start':
            p_lca.append(ancest[node])
            node = ancest[node]
  
        node, q_lca = q, [q]
        while ancest[node] != 'start':
            q_lca.append(ancest[node])
            node = ancest[node]
        
        for node in q_lca:
            if node in p_lca:
                return(node)

## using BST! 

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        
        while root:
            if p.val <= root.val <= q.val or q.val <= root.val <= p.val:
                return(root)
            elif p.val < root.val and q.val < root.val:
                root = root.left
            else:
                root = root.right
            
# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
