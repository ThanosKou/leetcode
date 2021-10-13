# With min heaps!

class Solution(object):
    def mergeKLists(self, lists):

        dummy = ListNode(0)
        curr = dummy
        
        to_be_sorted = [(head.val, i, head) for i, head in enumerate(lists) if head]
        heapify(to_be_sorted)


        while to_be_sorted != []:
            val, i, node = to_be_sorted[0]
            if not node.next:
                heappop(to_be_sorted)
            else:
                heapreplace(to_be_sorted, (node.next.val, i, node.next))
            curr.next = node
            curr = curr.next
        return(dummy.next)

# https://leetcode.com/problems/merge-k-sorted-lists/

    
