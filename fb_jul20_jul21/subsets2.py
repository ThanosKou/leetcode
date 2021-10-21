class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        
        res = []
        for mask in range(1<<len(nums)):
            cur_res = []
            for j in range(len(nums)):
                if (1<<j & mask) > 0:
                    cur_res.append(nums[j])
            if sorted(cur_res) not in res:
                res.append(sorted(cur_res))
        
        return(res)

# https://leetcode.com/problems/subsets-ii/


## Backtracking
