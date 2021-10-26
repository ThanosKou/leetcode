class Solution:
    def splitArray(self, nums: List[int], m: int) -> int:
        
        left, right = max(nums), sum(nums)
        while left < right:
            mid = left + (right - left)//2 
            if self.isSumLargeEnough(nums, m, mid):
                right = mid
            else:
                left = mid + 1
        return(left)
    
    
    def isSumLargeEnough(self, nums, m, trySum):
        
        array_ind = 0
        total_sum = 0
        for i in range(len(nums)):
            
            total_sum += nums[i]
            
            if total_sum > trySum:
                if array_ind == m - 1:
                    return(False)
                total_sum = nums[i]
                array_ind += 1
        
        return(total_sum <= trySum)


# https://leetcode.com/problems/split-array-largest-sum/
