class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        
        rs = [amount + 1]*(amount + 1)
        rs[0] = 0
        
        for i in range(1,amount + 1):
            for c in coins:
                if i>=c:
                    rs[i] = min(rs[i],1+rs[i-c])
        if rs[amount] == amount + 1:
            return(-1)
        return(rs[amount])
             
