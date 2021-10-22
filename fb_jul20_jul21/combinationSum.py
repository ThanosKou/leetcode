class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        
        candidates.sort()
        res = []
        self.dfs(candidates, res, [], target, 0)

        return(res)
    
    def dfs(self, nums, res, path, target, j):
        
        if sum(path) == target:
            res.append(path)
            j += 1
            return
        if sum(path) > target:
            return
        
        for i in range(len(nums)):
            j = i
            self.dfs(nums[j:], res, path + [nums[i]], target, j)
            
            
   


# https://leetcode.com/problems/combination-sum/
