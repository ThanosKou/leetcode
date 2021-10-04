class Solution:
    def rob(self, nums: List[int]) -> int:
        # Break the circle by considering array[0..n-1], [1..n]
        if len(nums)<=2:
            return(max(nums))
        
        def rob_house(left_ind,right_ind):
            
            numss = nums[left_ind:right_ind]
            cache = [0]*(len(numss))
            cache[0] = numss[0]
            cache[1] = max(numss[0],numss[1])
            for i in range(2,len(numss)):
                cache[i] = max(numss[i] + cache[i-2],cache[i-1])

            return(cache[-1])
        
        return(max(rob_house(0,len(nums)-1),rob_house(1,len(nums))))           
