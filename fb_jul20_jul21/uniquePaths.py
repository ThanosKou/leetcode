class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        
        self.directions = [[1,0], [0,1]]
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        cache = {}
        return(self.dfs(0, 0, obstacleGrid, m, n, cache))
 
    
    def isValid(self, x, y, obstacleGrid, m, n):
        if x >= 0 and x <= m-1 and y>=0 and y <= n-1 and obstacleGrid[x][y] == 0:
            return(True)
        return(False)
    
    def dfs(self, i, j, obstacleGrid, m, n, cache):
        if (i,j) in cache:
            return(cache[(i,j)])
        
        if not self.isValid(i, j, obstacleGrid, m, n):
            return(0)
        
        if i == m-1 and j == n-1:
            return(1)
        
        cache[(i,j)] = self.dfs(i, j+1, obstacleGrid, m, n, cache) + self.dfs(i+1, j, obstacleGrid, m, n, cache)
        return(cache[(i,j)])
      
  # DP
  
  class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        
        self.directions = [[1,0], [0,1]]
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        
        
        if obstacleGrid[0][0]:
            return(0)        
        
        dp = [[0 for _ in range(n)] for _ in range(m)]
        
        dp[0][0] = 1

            
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j]:
                    continue
                if i > 0: 
                    dp[i][j] += dp[i-1][j]
                if j > 0:
                    dp[i][j] += dp[i][j-1]     
        return(dp[m-1][n-1])

# https://leetcode.com/problems/unique-paths-ii/
