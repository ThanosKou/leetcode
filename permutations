class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result = []
        def rec(x,y):
            if len(y)==1:
                result.append(x+y)
            else:
                for i,v in enumerate(y):
                    rec(x+[v],y[:i]+y[i+1:])
                        
        rec([],nums)
        return(result)