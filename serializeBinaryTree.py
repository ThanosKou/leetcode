# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return('')
        queue = collections.deque([root])
        
        res = []
        while queue:
            node = queue.popleft()
            if not node:
                res.append('*')
            else:
                res.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
        return(','.join(res))
        
        
    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if data == '':
            return([])
        to_be_filled = data.split(',')
        
        root = TreeNode(int(to_be_filled[0]))
        queue = collections.deque([root])
        
        i = 1
        while queue:
            node = queue.popleft()
            if to_be_filled[i] != '*':
                node.left = TreeNode(int(to_be_filled[i]))
                queue.append(node.left)
            if to_be_filled[i+1] != '*':
                node.right = TreeNode(int(to_be_filled[i+1]))
                queue.append(node.right)
            i += 2
        return(root)
            
        
        

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))

# https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
