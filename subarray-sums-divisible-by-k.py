class Solution:
    def subarraysDivByK(self, A: List[int], K: int) -> int:
        
        d = collections.defaultdict(int)
        count = 0
        d[0]=1
        ans = csum = 0
        for val in A:
            csum = (csum + val)%K
            ans += d[csum%K]
            d[csum] +=1          
        return(ans)