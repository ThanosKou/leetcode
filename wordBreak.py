class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        
        #dp[i] means that s[0:i+1] is breakable, hence we start from s[0]
        dp = [False]*(len(s)+1)
        dp[0] = True
        for i in range(len(s)):
            if dp[i]: # no need to ckeck other i
                for j in range(i,len(s)):
                    if s[i:j+1] in wordDict:
                        dp[j+1] = True
        return(dp[-1])
