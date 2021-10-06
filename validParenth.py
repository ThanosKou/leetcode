class Solution:
    def isValid(self, s: str) -> bool:
        parenth_stack = []
        corresp = {")":"(", "}":"{", "]":"["}
        for char in s:
            if char not in corresp:
                parenth_stack.append(char)
            else:
                if not parenth_stack:
                    return(False)
                if parenth_stack[-1] == corresp.get(char):
                    parenth_stack.pop()
                else:
                    return(False)
        return(parenth_stack == [])
      
# https://leetcode.com/problems/valid-parentheses/
