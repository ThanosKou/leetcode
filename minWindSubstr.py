class Solution:
    def minWindow(self, s: str, t: str) -> str:
       
      from collections import Counter
  
      need, missing = Counter(t), len(t) # need shows how many occur of each character we need, missing indicates when valid substring
      I, J = 0, 0
      i = 0

      for j, ch in enumerate(s,1):
        if need[ch] > 0:
          missing -= 1
        need[ch] -= 1 # we don't care for negative appearances, it shows we dont need that character
        if not missing:
          while i < j and need[s[i]] < 0:
            need[s[i]] += 1
            i += 1
          if j - i <= J - I or not J:
              J, I = j, i

      return(s[I:J])
