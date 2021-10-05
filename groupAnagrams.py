def groupAnagrams(self, strs):
    d = {}
    for w in sorted(strs):
        key = tuple(sorted(w))
        d[key] = d.get(key, []) + [w]
    return d.values()
  
 class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        
        
        same_anagr_dict = {}
        for ind, st in enumerate(strs):
            sort_st = tuple(sorted(st))
            if sort_st not in same_anagr_dict:
                same_anagr_dict[sort_st] = [ind]
            else:
                same_anagr_dict[sort_st]. append(ind)
        
        res = []
        for st in same_anagr_dict:
            res_curr = []
            for ind in same_anagr_dict[st]:
                res_curr.append(strs[ind])
            res.append(res_curr)
        return(res)
