class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)-1
        
        
        while left <= right:
            mid = (left + right) >> 1
            if target == nums[mid]:
                return(mid)
            if nums[mid] >= nums[left]: #right-rotated
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else: #left-rotated
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return(-1)
