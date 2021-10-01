class Solution:
    def countBits(self, n: int) -> List[int]:
        
        output = []
        for i in range(n+1):
            count = 0
            while i:
                i &=(i-1)
                count += 1
            output.append(count)
        return(output)
      
class Solution:
    def countBits(self, n: int) -> List[int]:
        
        count = [0]*(n+1)
        for i in range(n+1):
            count[i] = count[i>>1] + i%2
        return(count)
