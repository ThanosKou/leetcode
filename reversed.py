class Solution:
    def reverse(self, x: int) -> int:
        if not x:
            return(x)
        sign = x/abs(x)
        x = abs(x)
        digits = []
        minv = -2**31
        maxv = 2**31-1
        while x != 0:
            digits.append(x%10)
            x = x // 10
        rev = 0
        for i,v in enumerate(digits):
            rev += v * 10**(len(digits)-i-1)
        rev = int(sign*rev)
        if minv <= rev <= maxv:
            return(rev)
        else:
            return(0)
            
