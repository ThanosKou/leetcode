class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if len(nums) == 1:
            return(nums)
        
        i = j = len(nums) - 1
        
        while i > 0 and nums[i-1] >= nums[i]:
            i -= 1
            
        if i == 0:
            nums.reverse()
            return
            
        pivot = i-1
        
        while nums[j] <= nums[pivot]:
            j -= 1
        
        nums[j], nums[pivot] = nums[pivot], nums[j]
        l, r = pivot+1, len(nums)-1  # reverse the second part
        
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1

# https://leetcode.com/problems/next-permutation/
