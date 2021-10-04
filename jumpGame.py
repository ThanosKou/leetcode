class Solution:
    def canJump(self, nums: List[int]) -> bool:
        
        lastGoodIndex = len(nums)-1
        for i in range(len(nums)-1,-1,-1):
            if i + nums[i] >= lastGoodIndex:
                lastGoodIndex = i
        
        return(lastGoodIndex == 0)
 # DP
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        
        dp = [0]*len(nums) # dp[i] is the furtherst we can reach from index i. If dp[i] == 0 at any point then False. If dp[i] == len(nums) - 1 then True
        dp[0] = nums[0]
        for i in range(1,len(nums)):
            dp[i] = max(i+nums[i],dp[i-1])
            if dp[i-1] < i:
                return(False)
            if dp[i] >= len(nums)-1:
                return(True)
        return(dp[len(nums)-2] >= len(nums)-1)
