class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        
        seen = {}
        for ind,val in enumerate(nums):
            if val in seen:
                ind_dist = seen[val]- ind
                if abs(ind_dist) <= k:
                    return(True)
            seen[val] = ind
        return(False)