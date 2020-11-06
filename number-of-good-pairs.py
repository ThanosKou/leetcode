class Solution:
    def numIdenticalPairs(self, nums: List[int]) -> int:
        
        # Dumb solution
        #pairs = 0
        #for i in range(len(nums)-1):
        #    for j in range(i+1,len(nums)):
        #        if nums[i] - nums[j] == 0:
        #            pairs += 1
        #return(pairs)
        
        # Better?         

        c = Counter(nums)
        pairs = 0
        for i in c:
            if c[i] > 1:
                pairs += c[i]*(c[i]-1)/2 #C(n,2)
        return(int(pairs))