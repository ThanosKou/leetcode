class Solution:
    #def subsets(self, nums: List[int]) -> List[List[int]]:
    #    res = [[]]
    #    for num in nums:
    #        res += [curr + [num] for curr in res]
    #        
    #    return(res)
    
    def subsets(self, nums: List[int]) -> List[List[int]]:
        
        res = []
        self.dfs(nums, [],res)
        return(res)
    
    def dfs(self, nums, path, res):
        res.append(path)
        for i in range(len(nums)):
            self.dfs(nums[i+1:], path + [nums[i]], res)
 

class Solution:    
    def subsets(self, nums: List[int]) -> List[List[int]]:
        
        output = []
        for mask in range((1<<len(nums))):
            cur_output = []
            for j in range(len(nums)):
                if 1<<j & mask > 0:
                    cur_output.append(nums[j])
            output.append(cur_output)
        
        return(output)

# https://leetcode.com/problems/subsets/
