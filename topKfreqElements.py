class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        
        most_comm = []
        l = Counter(nums).most_common(k)
        for i in range(len(l)):
            most_comm.append(l[i][0])
        
        return(most_comm)

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]: 
        # O(1) time 
        if k == len(nums):
            return nums
        
        count = Counter(nums)   
   
        return heapq.nlargest(k, count.keys(), key=count.get) 
  
  
# bucketsort
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]: 
        
        counts = Counter(nums).items()
        
        # first create frequency buckets
        
        buckets = [[] for _ in range(len(nums) + 1)]
        
        l = Counter(nums).items()

        buckets = [[] for _ in range(len(nums)+1)  ]

        for val, freq in l:
            buckets[freq].append(val)
        
        
        return(list(chain(*buckets))[::-1][:k])
      
      # https://leetcode.com/problems/top-k-frequent-elements/
