Class Solution:
        def frequencySort(self, nums: List[int]) -> List[int]:
        #from Collections import Counter
        c = Counter(nums)
        return(sorted(c.elements(), key=lambda n:(c[n],-n)))
