class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return(sorted(t)==sorted(s)) 
        
        #d = collections.defaultdict(int)
        #if len(t) == len(s):
        #    for i in range(len(t)):
        #        d[ord(t[i])-ord('a')] +=1
        #        d[ord(s[i])-ord('a')] -=1
        #    return(list(d.values())==[0]*len(d.values()))
        #else:
        #    return(False)
        