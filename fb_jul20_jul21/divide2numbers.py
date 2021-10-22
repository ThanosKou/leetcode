class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        

        neg = False
        if dividend == 0:
            return(0)
        if (dividend < 0 and divisor > 0) or (dividend > 0 and divisor < 0):
            neg = True
        dividend = abs(dividend)
        divisor = abs(divisor)
        
        
        quotient = 0
        cur_sum = divisor
        while cur_sum <= dividend:
            cur_quotient = 1
            while (cur_sum << 1) < dividend:
                cur_sum <<= 1
                cur_quotient <<= 1
            
            dividend -= cur_sum
            quotient += cur_quotient
            cur_sum = divisor
        
        return min(2147483647, max(-quotient if neg else quotient, -2147483648))


# https://leetcode.com/problems/divide-two-integers/
