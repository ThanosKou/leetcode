class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        d = {}
        for num in nums:
            if num in d:
                d[num] = d[num] + 1
            else:
                d[num] = 1
        for key,val in d.items():
            if val == 1:
                return(key)

    def singleNumber(self, nums: List[int]) -> int:
        
        return(2*sum(set(nums))-sum(nums))
