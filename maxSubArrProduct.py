class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        min_prev = max_prev = global_max = nums[0]
        
        for num in nums[1:]:
            curr_min = min(min_prev*num,max_prev*num,num)
            curr_max = max(min_prev*num,max_prev*num,num)
            global_max = max(global_max,curr_max)
            min_prev = curr_min
            max_prev = curr_max
            
        return(global_max)
