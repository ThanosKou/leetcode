class Solution:
    def isPalindrome(self, s: str) -> bool:
        new_st = ""
        for char in s:
            if char.isalnum():
                new_st += char.lower() 
        return(new_st =="".join(reversed(new_st)))
      
 # https://leetcode.com/problems/valid-palindrome/submissions/
