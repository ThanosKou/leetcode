class Solution:
    def isPalindrome(self, x: int) -> bool:
        st = str(x)
        rev = st[len(st)::-1]
        return (rev == st)
