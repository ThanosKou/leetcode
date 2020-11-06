class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        
        #if nums ==[] or len(nums)==1:
        #    return len(nums)
        #j = 1
        #for i in range(1,len(nums)):
        #    if nums[j] == nums[j-1]:
        #        del nums[j]
        #        j -= 1
        #    j += 1
        #return(len(nums))

        nums[:] = set(nums)
        nums.sort()
        return len(nums)