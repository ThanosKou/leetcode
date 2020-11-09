class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        if nums == []:
            return False
        d = Counter(nums)
        return(False if max(d.values())==1 else True)