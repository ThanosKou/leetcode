class Solution:
    def countPrimes(self, k:int) -> int:    
        if k < 3:
            return(0)
        isPrime = [True]*k
        isPrime[0] = isPrime[1] = False
        for i in range(2,int(k**0.5)+1):
            if isPrime[i]:
                for j in range(i*i,k,i): #e.g. for 7, start from 7*7=49
                    isPrime[j] = False
        return(sum(isPrime))
        
