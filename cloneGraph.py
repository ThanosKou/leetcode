# Overall approach is graph traversal. The trick is in cloning.

# Use a dictionary with node as key and the clone as value
# when visiting the neighbors, create a key-value pair for each child node.
# add that key value pair to the neighbor for the parent node if that neighboring node has not been visited.
# Ensuring that the neighboring node is not visited is crucial as it could lead to a spiral (i made that mistake)
# return the value of the node being passed in


"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""
# DFS recursively
class Solution:
    
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return
        dic = {}
        node_clone = Node(node.val, [])
        dic = {node: node_clone}
        self.dfs(node, dic)
        return(node_clone)
    
    def dfs(self, node, dic):
        for neigh in node.neighbors:
            if neigh not in dic:
                neighCopy = Node(neigh.val, [])
                dic[neigh] = neighCopy
                dic[node].neighbors.append(neighCopy)
                self.dfs(neigh, dic)
            else:
                dic[node].neighbors.append(dic[neigh])
                
                
    # DFS iterative
class Solution:
    
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return
        nodeCopy = Node(node.val, [])
        dic = {node: nodeCopy}
        stack = [node]
        while queue:
            curr = stack.pop()
            for neigh in curr.neighbors: 
                if neigh not in dic:
                    neighCopy = Node(neigh.val, [])
                    dic[neigh] = neighCopy
                    dic[curr].neighbors.append(neighCopy)
                    stack.append(neigh)
                else:
                    dic[curr].neighbors.append(dic[neigh])
        return(nodeCopy)
    
# BFS same as DFS iterative, but queue instead of stack
    class Solution:
    
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return
        nodeCopy = Node(node.val, [])
        dic = {node: nodeCopy}
        queue = collections.deque([node])
        while queue:
            curr = queue.popleft()
            for neigh in curr.neighbors: 
                if neigh not in dic:
                    neighCopy = Node(neigh.val, [])
                    dic[neigh] = neighCopy
                    dic[curr].neighbors.append(neighCopy)
                    queue.append(neigh)
                else:
                    dic[curr].neighbors.append(dic[neigh])
        return(nodeCopy)
    
                
