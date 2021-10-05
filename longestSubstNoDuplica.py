class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        
        dic, start, max_len = {}, 0, 0

        for ind,st in enumerate(s):
            if st in dic:
                max_len = max(ind - start,max_len)
                start = max(start,dic[st] + 1)

            dic[st] = ind
        
        return(max(max_len,len(s)-start))
