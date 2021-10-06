class Solution:
    def longestPalindrome(self, s: str) -> str:
        
        dp = [[False for i in range(len(s))] for j in range(len(s))]
        
        for i in range(len(s)):
            dp[i][i] = True
            longest = s[i]
            
        for i in range(len(s)-1,-1,-1):
            for j in range(i+1, len(s)):
                if s[i] == s[j]:
                    if j-i == 1 or dp[i+1][j-1]:
                        dp[i][j] = 1
                        if len(s[i:j+1]) > len(longest):
                            longest = s[i:j+1]
        
        return(longest)
 
# https://leetcode.com/problems/longest-palindromic-substring/
