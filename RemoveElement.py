class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        if nums == []:
            return 0
        j=0
        for i in range(len(nums)):
            if nums[j] == val:
                del nums[j]
                j = j-1
            j = j + 1
        return(len(nums))
                