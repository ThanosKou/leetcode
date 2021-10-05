class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        
        counts = collections.Counter()
        start, end = 0, 0 # window limits
        max_len = 0
        for end in range(len(s)):
            counts[s[end]] += 1
            if end - start + 1 > k + counts.most_common(1)[0][1]: # we want at most window length == k + most_common
                counts[s[start]] -= 1
                start += 1 # window shift
            max_len = max(max_len,end - start + 1)
        return(max_len)
