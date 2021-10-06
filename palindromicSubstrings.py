class Solution:
    def countSubstrings(self, s: str) -> int:
        
        dp = [[False for i in range(len(s))] for j in range(len(s))]
        count = 0
        for i in range(len(s)):
            dp[i][i] = True
            count += 1
        
        
        for i in range(len(s)-1,-1,-1):
            for j in range(i+1, len(s)):
                if s[i] == s[j]:
                    if j-i==1 or dp[i+1][j-1]:
                        dp[i][j] = True
                        count += 1
        return(count)
      
 # https://leetcode.com/problems/palindromic-substrings/
