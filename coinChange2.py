class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        
        dp = [0]*(amount +1)
        dp[0] = 1
        for coin in coins:
            for i in range(1,amount+1):
                if i>= coin:
                    dp[i] += dp[i-coin]
        return(dp[amount])

    # Difference betweem coinChange2 (no duplicates) and combination sum (duplicates matter:order matters): In the comb sum for each position you try each coin. 
    # In that way, you are distinguishing two different orderings. 
    # If you instead loop through every position for each coin, then you are imposing a specific ordering
