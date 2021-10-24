class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        
        
        dp = [False]*(len(s)+1)
        
        dp[0] = 1
        candidates = []
        
        for i in range(len(s)):
            if dp[i]:
                for j in range(i, len(s)):
                    if s[i:j+1] in wordDict:
                        dp[j+1] = True #everytthing up to j + 1 is breakable 
        if not dp[-1]:
            return([])
        
        pos = []
        for i in range(len(dp)):
            if dp[i]:
                pos.append(i-1)
                
        pos.pop(0)
        res = []
        self.build(pos, s, "", res, wordDict, 0)
        return res

    def build(self, pos, s, choice, res, wordDict, start):
        if start > pos[-1]:
            res.append(choice[:-1])
            return 

        for i in range(len(pos)):
            if s[start: pos[i]+1] in wordDict:
                self.build(pos, s, choice+s[start:pos[i]+1]+" ", res, wordDict, pos[i]+1)





#
