class Solution:
    def addBinary(self, a: str, b: str) -> str:
        
        res = []
        carry = 0
        a = list(a)
        b = list(b)
        
        while a or b or carry:
            
            if a:
                carry += int(a.pop())
            if b:
                carry += int(b.pop())
            
            res.append(str(carry%2))
            
            carry //= 2
            
        return(''.join(reversed(res)))

# https://leetcode.com/problems/add-binary/
