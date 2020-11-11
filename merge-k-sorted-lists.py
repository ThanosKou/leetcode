class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        
        ans = ListNode(0)
        point = ans
        arr = []
        
        for llist in lists:
            while llist:
                arr.append(llist.val)
                llist = llist.next
        
        for i in sorted(arr):
            point.next = ListNode(i)
            point = point.next
        
        return(ans.next)
		
	from queue import PriorityQueue

### Using prioritty queue 

# class Solution:
    # def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        

        # ans = ListNode(0)
        # point = ans
        # arr = []
        # q = PriorityQueue()
        
        # for llist in lists:
            # while llist:
                # q.put(llist.val)
                # llist = llist.next
        
        # while not q.empty():
            # q_val = q.get()
            # point.next = ListNode(q_val)
            # point = point.next
            
        # return(ans.next)
		
### Kinda like merge sort, merge lists by pairs  using merge2Lists 		
		
# class Solution(object):
    # def mergeKLists(self, lists):
        # """
        # :type lists: List[ListNode]
        # :rtype: ListNode
        # """
        # amount = len(lists)
        # interval = 1
        # while interval < amount:
            # for i in range(0, amount - interval, interval * 2):
                # lists[i] = self.merge2Lists(lists[i], lists[i + interval])
            # interval *= 2
        # return lists[0] if amount > 0 else None  
        
    # def merge2Lists(self, list1, list2):
        # """
        # :type list1: List[ListNode]
        # :type list2: List[ListNode]
        # :rtype: ListNode
        # """      
        # ans = ListNode(0)
        # point = ans
        
        # while list1 and list2:
            # if list1.val < list2.val:
                # point.next = list1
                # list1 = list1.next
            # else:
                # point.next = list2
                # list2 = list2.next
            # point = point.next
                
        # if not list1:
            # point.next = list2
        # if not list2:
            # point.next = list1
                
        # return(ans.next)
    