class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        
        
        self.directions = [[-1,0], [0,-1], [0,1], [1,0]]
        m = len(heights)
        n = len(heights[0])
        p_visited = [[False for _ in range(n)] for _ in range(m)]
        a_visited = [[False for _ in range(n)] for _ in range(m)]
        res = []
        for i in range(m):
            self.dfs(heights, i, 0, m, n, p_visited)
            self.dfs(heights, i, n-1, m, n, a_visited)
        
        for j in range(n):
            self.dfs(heights, 0, j, m, n, p_visited)
            self.dfs(heights, m-1, j, m, n, a_visited)
        
        for i in range(m):
            for j in range(n):
                if p_visited[i][j] and a_visited[i][j]:
                    res.append([i, j])
        return(res)
        
    def dfs(self, heights, i, j, m, n, visited):
        visited[i][j] = True
        for direct in self.directions:
            x, y = i + direct[0], j + direct[1]
            if x < 0 or x >= m or y < 0 or y >= n or heights[x][y] < heights[i][j] or visited[x][y]: # we don't want that
                continue
                
            self.dfs(heights, x, y, m, n, visited)
        
# https://leetcode.com/problems/pacific-atlantic-water-flow/
