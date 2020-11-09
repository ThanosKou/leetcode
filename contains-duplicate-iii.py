class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:
        if t<0 or k<1:
            return(False)
        bucket = collections.OrderedDict()  
        for n in nums:
            key = n//(t or 1) # hashing in buckets [x//t - 1, x//t + 1] 
            for b in [bucket.get(key-1), bucket.get(key), bucket.get(key+1)]:
                if b is not None and abs(b-n)<=t: #take care of complement n=9,t=3, 14//3=3 but 14-9>3
                    return(True)                   
            if len(bucket)==k: # pop an item to take care of max k diference 
                bucket.popitem(False)
            bucket[key] = n
        return(False)