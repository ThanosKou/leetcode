class Solution:
    def minWindow(self, s: str, t: str) -> str:
        
        need = collections.Counter(t)
        missing = sum([i for i in need.values()])
        j = 0
        min_len = len(s)+1
        
        j = 0
        start, end = 0, 0
        for i in range(len(s)):
            
            if need[s[i]] > 0:
                missing -= 1    
            need[s[i]] -= 1
            
            while missing == 0:
                if i - j + 1 < min_len:
                    min_len = i - j + 1
                    start = j
                    end = i+1
                need[s[j]] += 1
                if need[s[j]] > 0:
                    missing += 1            
                j += 1
        
        return(s[start:end])

#  https://leetcode.com/problems/minimum-window-substring/
