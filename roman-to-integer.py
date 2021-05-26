class Solution:
    def romanToInt(self, s: str) -> int:
        
        if not s:
            return(s)
        
        dic = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
        num = 0
        for ind,st in enumerate(s):  
            if ind < len(s)-1:
                if st == 'I' and (s[ind+1] == 'V' or s[ind+1] == 'X'):
                    num -= 1
                elif st == 'X' and (s[ind+1] == 'L' or s[ind+1] == 'C'):
                    num -= 10
                elif st == 'C' and (s[ind+1] == 'D' or s[ind+1] == 'M'):
                    num -= 100
                else:
                    num += dic[st]
            else:
                num += dic[st]
        return(num)

        
