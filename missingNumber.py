class Solution:
    def missingNumber(self, nums: List[int]) -> int: # a xor a xor b = b
        
        res = 0
        for i in range(len(nums)): 
            res ^= i
            res ^= nums[i]
        
        return(res^(i+1))
      
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        
        summ = 0
        for i in range(len(nums)):
            summ += nums[i] - i
        
        return(-(summ - len(nums)))
