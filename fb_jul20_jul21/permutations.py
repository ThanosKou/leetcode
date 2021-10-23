class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        k = len(nums)
        self.dfs(nums, res, [], k)
        return(res)
    
    def dfs(self, nums, res, path, k):
        if len(path) == k:
            res.append(path)
            return
        
        for i in range(len(nums)):
            if nums[i] not in path:
                self.dfs(nums, res, path + [nums[i]], k)

#  https://leetcode.com/problems/permutations/
