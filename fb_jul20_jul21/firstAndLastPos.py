class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        
        if target not in nums:
            return([-1,-1])
        
        notFound = True
        left = 0
        right = len(nums)-1
        while left <= right: 
            mid = left + (right - left)//2
                
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = right = mid 
                while left >= 0 and nums[mid] == nums[left] :
                    left -= 1
                
                while right <= len(nums) -1 and nums[mid] == nums[right]:
                    right += 1
        
                return(left+1, right-1)



# https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
