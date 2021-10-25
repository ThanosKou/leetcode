class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        
        if not matrix: return 0
    
        self.directions = [[0,-1], [-1,0], [0,1], [1,0]]
        m = len(matrix)
        n = len(matrix[0])
        cache = [[-1 for _ in range(n)] for _ in range(m)]   
        
        res = 0
        for i in range(m):
            for j in range(n):
                if cache[i][j] == -1:
                    res = max(res, self.dfs(i, j, matrix, cache, m, n))
        return(res)
        
        
    def dfs(self, i, j, matrix, cache, m, n):
        if cache[i][j] != -1:
            return(cache[i][j])
        res = 1
        for direc in self.directions:
            x, y = i + direc[0], j + direc[1]
            if x >= 0 and x <= m-1 and y >= 0 and y <= n-1 and matrix[x][y] > matrix[i][j]:
                path_len = 1 + self.dfs(x, y, matrix, cache, m, n)
                res = max(res, path_len)
            else:
                continue
        cache[i][j] = res
        return(res)

# https://leetcode.com/problems/longest-increasing-path-in-a-matrix/
