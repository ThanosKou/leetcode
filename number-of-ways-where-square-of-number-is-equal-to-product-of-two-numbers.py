class Solution:
    def numTriplets(self, nums1: List[int], nums2: List[int]) -> int:
               
        if nums1 == [] or nums2 == []:
            return 0
            
        n1_squared = Counter([a*a for a in nums1])
        n2_squared = Counter([a*a for a in nums2])
        
        triplets = 0

        for i in range(len(nums1)-1):
            for j in range(i+1,len(nums1)):
                v = nums1[i]*nums1[j]
                if n2_squared.get(v):
                    triplets += n2_squared[v]

        for i in range(len(nums2)-1):
            for j in range(i+1,len(nums2)):
                v = nums2[i]*nums2[j]
                if n1_squared.get(v):
                    triplets += n1_squared[v]
        return(triplets)