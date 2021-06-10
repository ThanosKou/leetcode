class Solution:
    def isHappy(self, n: int) -> bool:
        sum_dic = [n]
        sum = 0
        while sum not in sum_dic:
            sum_dic.append(sum)
            sum = 0
            while n:
                sum += (n%10)**2
                n = n//10
            if sum == 1:
                return(True)
            else:
                n = sum
        return(False)

    #class Solution:
    #def isHappy(self, n: int) -> bool:
    #    visited = set()
    #    while n != 1 and not n in visited:
    #        visited.add(n)
    #        n = sum(map(lambda x:int(x)**2, str(n))) #str(n) can be used as iterbale!
    #    return not n in visited
