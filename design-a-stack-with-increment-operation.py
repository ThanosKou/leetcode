class CustomStack:

    def __init__(self, maxSize: int):
        self.stack = []
        self.maxsize = maxSize



    def push(self, x: int) -> None:
        if len(self.stack) < self.maxsize:
            self.stack.append(x)
 
        

    def pop(self) -> int:
        if self.stack:
            return(self.stack.pop())
        else:
            return(-1)
        

    def increment(self, k: int, val: int) -> None:
        if len(self.stack) <= k:
            for elem in range(len(self.stack)):
                self.stack[elem] = self.stack[elem] + val
        else:
            for elem in range(k):
                self.stack[elem] = self.stack[elem] + val
        