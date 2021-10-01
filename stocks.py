class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        
        peak = 0
        low = 0 
        profit = 0
        for i,v in enumerate(prices):
            if v < prices[low]:
                low = i
                peak = i
            if v > prices[peak]:
                peak = i
            profit = max(profit,prices[peak]-prices[low])
        return(profit)

    # Kradane algorithm
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        ans = 0
        curSum = 0
        for i in range(n-1):
            curSum += prices[i+1] - prices[i]
            if curSum < 0:
                curSum = 0
            ans = max(ans, curSum)
        return ans
