class Solution:
    def canArrange(self, arr: List[int], k: int) -> bool:
        
        from collections import Counter
        cnt = Counter([i%k for i in arr])
        for i in cnt:
            if i==0:
                if cnt[i]%2 != 0:
                    return(False)
            elif cnt[i] != cnt[k-i]:
                return(False)
        return(True)
        