class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        
        
        dp = [1 for i in range(m)]
        l = 1
        for i in range(1,n):
            for j in range(1,m):
                # dp[i][j] = dp[i][j-1] + dp[i-1][j] =>
                # dp[j] = dp[j-1] + dp[j] =>
                dp[j] += dp[j-1]
        return(dp[-1])
