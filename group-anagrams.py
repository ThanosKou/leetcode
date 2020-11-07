class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        
        ans=collections.defaultdict(list)
        for st in strs:
            ans[tuple(sorted(st))].append(st)
        return(ans.values())
        
        
        ## OR we could use hashing from str to numbers ord(s)-ord('a')