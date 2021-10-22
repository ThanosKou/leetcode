class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        
        seen = {}
        
        left = 0
        right = 1
        cur_max = 0
        for st in s:
            if st not in seen or seen[st] == 0:
                seen[st] = 1
                right += 1
            else: # duplicate
                while s[left] != st:
                    seen[s[left]] -= 1
                    left += 1
                left += 1
                right += 1
            cur_max = max(right - left - 1, cur_max)
        
        return(cur_max)


## better tho

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        
        dic, start, max_len = {}, 0, 0

        for ind,st in enumerate(s):
            if st in dic:
                max_len = max(ind - start,max_len)
                start = max(start,dic[st] + 1) # care about start! as you assign duplicate char as new start, you may lose others in between

            dic[st] = ind
        
        return(max(max_len,len(s)-start))










# https://leetcode.com/problems/longest-substring-without-repeating-characters/
