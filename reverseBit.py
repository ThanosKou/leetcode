class Solution:
    def reverseBits(self, n: int) -> int:
        
        mask = 1 << 31
        res = 0
        for i in range(32):
            res << 1
            if n&1:
                res|= mask
            n >>= 1
            mask >>= 1
        return(res)
