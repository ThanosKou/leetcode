class Solution:
    def hammingWeight(self, n: int) -> int:
        
        count = 0
        mask = 1
        for i in range(32):
            if n&mask:
                count += 1
            mask <<= 1
        return(count)
      
      
class Solution:
    def hammingWeight(self, n: int) -> int:
        
        count = 0
        while n:
            n = n&(n-1)
            count += 1
        return(count)
