#Backtracking

class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        
        nums = [i for i in range(1,n+1)]
        
        res = []
        self.dfs(nums, k, [], res)
        return(res)
        
    def dfs(self, nums, k, path, res):
        
        if len(path) == k:
            res.append(path)
            return
        for i in range(len(nums)):
            self.dfs(nums[i+1:], k, path + [nums[i]], res)

# https://leetcode.com/problems/combinations/
