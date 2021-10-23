# backtracking 
    
    class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        
        candidates.sort()
        res = []
        self.dfs(candidates, target, res, [])
        return(res)
        
    def dfs(self, nums, target, res, path):
        if target == 0:
            res.append(sorted(path))
            return
        
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]: ## this is crucial
                continue           
            if nums[i] > target:
                break
            self.dfs(nums[i+1:], target - nums[i], res, path + [nums[i]])

# https://leetcode.com/problems/combination-sum-ii/
