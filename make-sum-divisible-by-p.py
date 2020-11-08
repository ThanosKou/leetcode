class Solution:
    def minSubarray(self, nums: List[int], p: int) -> int:
        need = sum(nums)%p
        if need == 0:
            return(0)
        
        d = {0:-1}
        ans = len(nums)
        csum = 0
        for ind,val in enumerate(nums):
            csum = (csum + val)%p
            d[csum] = ind
            if (csum - need)%p in d:
                ans = min(ans, ind-d[(csum -need)%p])
        return(ans if ans<len(nums) else -1)
        
        