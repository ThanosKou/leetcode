class MedianFinder:

    def __init__(self):
        self.left = []
        self.right = []
        heapq.heapify(self.left)
        heapq.heapify(self.right)

    def addNum(self, num: int) -> None:
        if not self.left and not self.right:
            heapq.heappush(self.right,num)
            return()
        if num > self.findMedian():
            heapq.heappush(self.right,num)
        else:
            heapq.heappush(self.left,-num)
            
        # now let's check for re-calibration
        if abs(len(self.left) - len(self.right)) > 1:
            if len(self.left) > len(self.right):
                k = -heapq.heappop(self.left)
                heapq.heappush(self.right, k)
            else:
                k = heapq.heappop(self.right)
                heapq.heappush(self.left, -k)

    def findMedian(self) -> float:

        if len(self.left) > len(self.right):
            return(-self.left[0])
        elif len(self.left) < len(self.right):
            return(self.right[0])
        else:
            return((self.right[0] - self.left[0])/2 )

        
        # a cool solution! with small and large heaps. 2 scenarios: 1) small_size=large_size, 2) large_size = small + 1
from heapq import *


class MedianFinder:
    def __init__(self):
        self.small = []  # the smaller half of the list, max heap (invert min-heap)
        self.large = []  # the larger half of the list, min heap

    def addNum(self, num):
        if len(self.small) == len(self.large):
            heappush(self.large, -heappushpop(self.small, -num))
        else:
            heappush(self.small, -heappushpop(self.large, num))

    def findMedian(self):
        if len(self.small) == len(self.large):
            return float(self.large[0] - self.small[0]) / 2.0
        else:
            return float(self.large[0])
# https://leetcode.com/problems/find-median-from-data-stream/
