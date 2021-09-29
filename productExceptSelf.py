class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        p = 1
        output = []
        for i,v in enumerate(nums):
            output.append(p)
            p *= v
        p = 1
        for i in range(len(nums)-1,-1,-1):
            output[i] *= p
            p*=nums[i]
            
        return(output)
