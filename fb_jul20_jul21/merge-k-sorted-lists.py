# With min heaps!

class Solution(object):
    def mergeKLists(self, lists):
        from heapq import heapify, heappush
        min_heap = []
        heapify(min_heap)
        
        for i in range(len(lists)):
            if lists[i]:
                heappush(min_heap, [lists[i].val,i])
                lists[i] = lists[i].next
        
        dummy = ListNode(0)
        curr = dummy
        while min_heap:
            curr_min, ind = heappop(min_heap)
            curr.next = ListNode(curr_min)
            curr = curr.next    
            
            if lists[ind]:
                heappush(min_heap, [lists[ind].val, ind])
                lists[ind] = lists[ind].next
        
        return(dummy.next)

# https://leetcode.com/problems/merge-k-sorted-lists/
