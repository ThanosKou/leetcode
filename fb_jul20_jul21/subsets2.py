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
    
## Backtracking
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        
        res = []
        nums.sort()
        self.dfs(nums, res, [])
        return(res)
    
    
    def dfs(self, nums, res, path):
        res.append(path)        
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            self.dfs(nums[i+1:], res, path + [nums[i]])
            
            
       # https://leetcode.com/problems/subsets-ii/ 
