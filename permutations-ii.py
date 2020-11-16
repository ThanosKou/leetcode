class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        
        res = []
        def rec(x,y):
            if len(y)==1:
                if x+y not in res:
                    res.append(x+y)
            else:
                for i,v in enumerate(y):
                    rec(x+[v],y[:i] + y[i+1:])

        rec([],nums)    
        return(res)
        