class Solution:    
    def isValid(self, x, y, grid):
        m = len(grid)
        n = len(grid[0])
        if x < 0 or y < 0 or x >= m or y >= n:
            return False
        return True
    
    def numIslands(self, grid: List[List[str]]) -> int:
        
        if not grid or not grid[0]:
            return(0)
        # need to find number of DAGs
        
        self.directions = [[-1,0], [0,-1], [1,0], [0,1]]
        m = len(grid)
        n = len(grid[0])
        visited = [[False for _ in range(n)] for _ in range(m)]
        
        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1'  and (not visited[i][j]):
                    count += 1
                    self.dfs(grid, i, j, visited)
        return(count)
            
            
    def dfs(self, grid, i, j, visited):
        visited[i][j] = True
        for direc in self.directions:
            x, y = i + direc[0], j + direc[1]
            if self.isValid(x,y,grid) and grid[x][y] == '1' and (not visited[x][y]):
                self.dfs(grid, x, y, visited) 
# Instead of visited, we could instead change the grid[][] itself. Whenever we visit a node, we can do grid[i][j] = '*'
# and in the if condition of the recursion, ask grid[x][y] == '1'
                
                
# https://leetcode.com/problems/number-of-islands/
