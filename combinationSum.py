class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        
        dp = [0]*(target + 1)
        for num in nums:
            if num <= target:
                dp[num] = 1
        for i in range(target+1):
            for num in nums:
                if i>num:
                    dp[i] += dp[i-num]
        return(dp[-1])
