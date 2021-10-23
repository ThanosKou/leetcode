class Solution:
    def simplifyPath(self, path: str) -> str:
        
        
        separate = path.split('/')
        
        stack = []
        for direc in separate:
            if direc == '.' or direc == '':
                continue
            if direc == '..':
                if stack:
                    stack.pop()
                continue
            stack.append(direc)
        
        
        return('/'+'/'.join(stack))

# https://leetcode.com/problems/simplify-path/
