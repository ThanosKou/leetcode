# Similar to given an array of positivie numbers, find the maximum sub of non-adjacent numbers
class Solution:
    def rob(self, nums: List[int]) -> int:
        
        cache = [0]*len(nums)
        cache[0] = nums[0]
        if len(nums)<=2:
            return(max(nums))
        cache[1] = max(nums[0],nums[1])
                   
        for i in range(2,len(nums)):
            cache[i] = max(nums[i] + cache[i-2],cache[i-1])
        
        return(cache[-1])
                   
