class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if not nums:
            return(nums)
        output = []
        nums1 = sorted(nums)
        for ind,target in enumerate(nums1):
            if ind > 0 and nums1[ind] == nums1[ind-1]:
                continue
            dic = {}
            for i,v in enumerate(nums1):
                if i != ind:
                    if v not in dic:
                        dic[-target - v] = i
                    else:
                        output.append([target,v,-target -v])
        out_set = set(tuple(sorted(x)) for x in output)
        output = list(out_set)
        return(output)
            
