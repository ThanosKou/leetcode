    class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        
        dp = [0]*(target + 1)
        for num in nums:
            if num <= target:
                dp[num] = 1
        for i in range(target+1):
            for num in nums:
                if i>num:
                    dp[i] += dp[i-num]
        return(dp[-1])

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
                if v in n2_squared:
                    triplets += n2_squared[v]

        for i in range(len(nums2)-1):
            for j in range(i+1,len(nums2)):
                v = nums2[i]*nums2[j]
                if v in n1_squared: 
                    triplets += n1_squared[v]
        return(triplets)

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        
        peak = 0
        low = 0 
        profit = 0
        for i,v in enumerate(prices):
            if v < prices[low]:
                low = i
                peak = i
            if v > prices[peak]:
                peak = i
            profit = max(profit,prices[peak]-prices[low])
        return(profit)

    # Kradane algorithm
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        ans = 0
        curSum = 0
        for i in range(n-1):
            curSum += prices[i+1] - prices[i]
            if curSum < 0:
                curSum = 0
            ans = max(ans, curSum)
        return ans

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        
        lastGoodIndex = len(nums)-1
        for i in range(len(nums)-1,-1,-1):
            if i + nums[i] >= lastGoodIndex:
                lastGoodIndex = i
        
        return(lastGoodIndex == 0)
 # DP
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        
        dp = [0]*len(nums) # dp[i] is the furtherst we can reach from index i. If dp[i] == 0 at any point then False. If dp[i] == len(nums) - 1 then True
        dp[0] = nums[0]
        for i in range(1,len(nums)):
            dp[i] = max(i+nums[i],dp[i-1])
            if dp[i-1] < i:
                return(False)
            if dp[i] >= len(nums)-1:
                return(True)
        return(dp[len(nums)-2] >= len(nums)-1)

class Solution:
    def isPalindrome(self, x: int) -> bool:
        st = str(x)
        rev = st[len(st)::-1]
        return (rev == st)

class Solution:
    def missingNumber(self, nums: List[int]) -> int: # a xor a xor b = b
        
        res = 0
        for i in range(len(nums)): 
            res ^= i
            res ^= nums[i]
        
        return(res^(i+1))
      
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        
        summ = 0
        for i in range(len(nums)):
            summ += nums[i] - i
        
        return(-(summ - len(nums)))

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        
        stack = [root]
        while stack:
            curr = stack.pop()
            if curr:
                curr.left, curr.right  = curr.right, curr.left
                stack.append(curr.left)
                stack.append(curr.right)
        return(root)

# https://leetcode.com/problems/invert-binary-tree/

class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        if nums == []:
            return 0
        j=0
        for i in range(len(nums)):
            if nums[j] == val:
                del nums[j]
                j = j-1
            j = j + 1
        return(len(nums))
                
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        cnt = summ = 0
        d = collections.defaultdict(int)
        d[0] = 1
        for num in nums:
            summ += num
            cnt += d[summ - k]
            d[summ] += 1
        return cnt
class Solution:
    def destCity(self, paths: List[List[str]]) -> str:
        
        from collections import Counter
        c_arriving = Counter([city[1] for city in paths])
        c_leaving = Counter([city[0] for city in paths])
        for city in c_arriving:
            if city in c_arriving and city not in c_leaving:
                return(city)
            
            
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next 
class Solution:
    def nextLargerNodes(self, head: ListNode) -> List[int]:
        
        #while cur is not None:
        #    cur = cur.next
        #    l += 1
        stack = []
        ans = []
        ind = 0
        cur = head
        while cur is not None:
            ans.append(0)
            
            while stack and cur.val > stack[-1][0]: # as soon as I find a larger guy I stop
                _,index = stack.pop()
                ans[index] = cur.val
                
            stack.append((cur.val,ind))
            ind += 1
            cur = cur.next
        return(ans)
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        
        mapping_s_t = {}
        mapping_t_s = {}
        
        for c1, c2 in zip(s, t):
            
            # Case 1: No mapping exists in either of the dictionaries
            if (c1 not in mapping_s_t) and (c2 not in mapping_t_s):
                mapping_s_t[c1] = c2
                mapping_t_s[c2] = c1
            
            # Case 2: Ether mapping doesn't exist in one of the dictionaries or Mapping exists and
            # it doesn't match in either of the dictionaries or both            
            elif mapping_s_t.get(c1) != c2 or mapping_t_s.get(c2) != c1:
                return False
            
        return True
            
            
        

def groupAnagrams(self, strs):
    d = {}
    for w in sorted(strs):
        key = tuple(sorted(w))
        d[key] = d.get(key, []) + [w]
    return d.values()
  
 class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        
        
        same_anagr_dict = {}
        for ind, st in enumerate(strs):
            sort_st = tuple(sorted(st))
            if sort_st not in same_anagr_dict:
                same_anagr_dict[sort_st] = [ind]
            else:
                same_anagr_dict[sort_st]. append(ind)
        
        res = []
        for st in same_anagr_dict:
            res_curr = []
            for ind in same_anagr_dict[st]:
                res_curr.append(strs[ind])
            res.append(res_curr)
        return(res)

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        
        ans=collections.defaultdict(list)
        for st in strs:
            ans[tuple(sorted(st))].append(st)
        return(ans.values())
        
        
        ## OR we could use hashing from str to numbers ord(s)-ord('a')

class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        
        dp = [[0 for j in range(len(text2)+1)] for i in range(len(text1)+1)]
        # dp[i][j] is the maximum common subsequence length for text1[:i] and text2[:j]
 
        for i in range(len(text1)):
            for j in range(len(text2)):
                if text1[i] == text2[j]:
                    dp[i+1][j+1] = 1 + dp[i][j]
                else:
                    dp[i+1][j+1] = max(dp[i][j+1],dp[i+1][j])
        return(dp[-1][-1])


#If we have two strings, say "nematode knowledge" and "empty bottle"
#To find the longest common subsequence, look at the first entry L[0,0]. 
#This is 7, telling us that the sequence has seven characters. 
#L[0,0] was computed as max(L[0,1],L[1,0]), corresponding to the subproblems formed by deleting either the "n" from the first string or the "e" from the second. 
#Deleting the "n" gives a subsequence of length L[0,1]=7, but deleting the "e" only gives L[1,0]=6, so we can only delete the "n".
#Now let's look at the entry L[0,1] coming from this deletion. A[0]=B[1]="e" so we can safely include this "e" as part of the subsequence, and move to L[1,2]=6. 
#Similarly this entry gives us an "m" in our sequence. 
#Continuing in this way (and breaking ties as in the algorithm above, by moving down instead of across) gives the common subsequence "emt ole".
#So we can find longest common subsequences in time O(mn)

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        if nums == []:
            return False
        d = Counter(nums)
        return(False if max(d.values())==1 else True)
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        
        
        self.directions = [[-1,0], [0,-1], [0,1], [1,0]]
        m = len(heights)
        n = len(heights[0])
        p_visited = [[False for _ in range(n)] for _ in range(m)]
        a_visited = [[False for _ in range(n)] for _ in range(m)]
        res = []
        for i in range(m):
            self.dfs(heights, i, 0, m, n, p_visited)
            self.dfs(heights, i, n-1, m, n, a_visited)
        
        for j in range(n):
            self.dfs(heights, 0, j, m, n, p_visited)
            self.dfs(heights, m-1, j, m, n, a_visited)
        
        for i in range(m):
            for j in range(n):
                if p_visited[i][j] and a_visited[i][j]:
                    res.append([i, j])
        return(res)
        
    def dfs(self, heights, i, j, m, n, visited):
        visited[i][j] = True
        for direct in self.directions:
            x, y = i + direct[0], j + direct[1]
            if x < 0 or x >= m or y < 0 or y >= n or heights[x][y] < heights[i][j] or visited[x][y]: # we don't want that
                continue
                
            self.dfs(heights, x, y, m, n, visited)
        
# https://leetcode.com/problems/pacific-atlantic-water-flow/

class Solution:
	def findKthPositive(self, arr: List[int], k: int) -> int:
		d = {}
		for num in arr:
			d[num] = True
		
		i = 1
		while k > 0:
			if i not in d:
				k -= 1
				missing = i
			i += 1
		return(missing)
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        
        #dp[i] means that s[0:i+1] is breakable, hence we start from s[0]
        dp = [False]*(len(s)+1)
        dp[0] = True
        for i in range(len(s)):
            if dp[i]: # no need to ckeck other i
                for j in range(i,len(s)):
                    if s[i:j+1] in wordDict:
                        dp[j+1] = True
        return(dp[-1])

class Solution:
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs: return ""
        if len(strs) == 1: return strs[0]
        
        strs.sort()
        p = ""
        for x, y in zip(strs[0], strs[-1]):
            if x == y: p+=x
            else: break
        return p

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        
        
        dp = [1 for i in range(m)]
        l = 1
        for i in range(1,n):
            for j in range(1,m):
                # dp[i][j] = dp[i][j-1] + dp[i-1][j] =>
                # dp[j] = dp[j-1] + dp[j] =>
                dp[j] += dp[j-1]
        return(dp[-1])

class Solution:
    def climbStairs(self, n: int) -> int:
        
        if n==1:
            return(1)
        ways = [0]*(n+1)
        ways[1] = 1
        ways[2] = 2
        for i in range(3,n+1):
            ways[i] = ways[i-1] + ways[i-2]
        return(ways[n])

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        res = []
        self.inOrder(root, res)
        
        return(res[k-1])
            
    
    def inOrder(self, root, res):
        if not root:
            return
        
        self.inOrder(root.left, res)
        res.append(root.val)
        self.inOrder(root.right, res)
        

# https://leetcode.com/problems/kth-smallest-element-in-a-bst/

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
class Solution:
    def longestPalindrome(self, s: str) -> str:
        
        dp = [[False for i in range(len(s))] for j in range(len(s))]
        
        for i in range(len(s)):
            dp[i][i] = True
            longest = s[i]
            
        for i in range(len(s)-1,-1,-1):
            for j in range(i+1, len(s)):
                if s[i] == s[j]:
                    if j-i == 1 or dp[i+1][j-1]:
                        dp[i][j] = 1
                        if len(s[i:j+1]) > len(longest):
                            longest = s[i:j+1]
        
        return(longest)
 
# https://leetcode.com/problems/longest-palindromic-substring/

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        current, previous = head, None
        
        while current:
            next = current.next 
            current.next = previous
            previous = current
            current = next
        return(previous) 
        
# and with recursion

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
 
        if not head or not head.next:
            return(head)

        rest = self.reverseList(head.next)
        head.next.next = head
        head.next = None

        return(rest)
        
 # https://leetcode.com/problems/reverse-linked-list/

# my solution was a bit dumb, I didnt use the fact that this is a BST, it can work with any binary tree

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        
        ancest = {root: 'start'}
        
        queue = collections.deque([root])
        while queue:
            node = queue.popleft()
            if node.left:
                ancest[node.left] = node
                queue.append(node.left)
            if node.right:
                ancest[node.right] = node
                queue.append(node.right)
        
        node, p_lca = p, [p]
        while ancest[node] != 'start':
            p_lca.append(ancest[node])
            node = ancest[node]
  
        node, q_lca = q, [q]
        while ancest[node] != 'start':
            q_lca.append(ancest[node])
            node = ancest[node]
        
        for node in q_lca:
            if node in p_lca:
                return(node)

## using BST! 

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        
        while root:
            if p.val <= root.val <= q.val or q.val <= root.val <= p.val:
                return(root)
            elif p.val < root.val and q.val < root.val:
                root = root.left
            else:
                root = root.right
            
# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/

class Solution:
    def minSubarray(self, nums: List[int], p: int) -> int:
        need = sum(nums)%p
        if need == 0:
            return(0)
        
        d = {0:-1}
        ans = len(nums)
        csum = 0
        for ind,val in enumerate(nums):
            csum = (csum + val)%p
            d[csum] = ind
            if (csum - need)%p in d:
                ans = min(ans, ind-d[(csum -need)%p])
        return(ans if ans<len(nums) else -1)
        
        
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        try:
            slow = head
            fast = head.next # to take care of single element case 
            while fast != slow:
                slow = slow.next
                fast = fast.next.next
            return(True)
        except:
            return(False)
            
# https://leetcode.com/problems/linked-list-cycle/

# 1) DFS topo sort

class Solution:
        
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        
        if not prerequisites:
            return(True)
        
        graph = {i: set() for i in range(numCourses)}
        
        for course_pair in prerequisites:
            next_course, prereq = course_pair
            graph[next_course].add(prereq) # doesnt matter where to put next and prereq
        
        state = [0]*numCourses
        
        def hasCycle(v):
            if state[v] == -1:
                return(True)
            if state[v] == 1:
                return(False)
            state[v] = -1
            for adj in graph[v]:
                if hasCycle(adj):
                    return(True)
            state[v] = 1
            return(False)

        for course in graph:
            if hasCycle(course):
                return(False)       
        return(True)
      
      # 2) BFS topo sort , kahn's algo
        
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        
        if not prerequisites:
            return(True)
        
        
        graph = {i: set() for i in range(numCourses)}
        in_degree = [0]*numCourses
        
        for course_pair in prerequisites: # create graph
            next_course, prereq_course = course_pair
            graph[next_course].add(prereq_course)
            in_degree[prereq_course] += 1
        
        queue = collections.deque()
        for i in range(len(graph)):
            if in_degree[i] == 0:
                queue.append(i)
                
        visited = set()
        
        while queue:
            curr = queue.popleft()
            visited.add(curr)
            for adj in graph[curr]:
                in_degree[adj] -= 1
                if in_degree[adj] == 0:
                    queue.append(adj)
          
        return(len(visited) == numCourses)

    # 3) DFS with stack 
        
class Solution:
        
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        
        if not prerequisites:
            return(True)
        
        graph = {i: set() for i in range(numCourses)}
        
        for course_pair in prerequisites:
            next_course, prereq = course_pair
            graph[prereq].add(next_course)
        

        visited = set()
        
            
        def hasCycle(v, stack):
            if v in visited:
                if v in stack: # it's like state == -1
                    return(True)
                return(False) # its like state == 1
            visited.add(v)
            stack.append(v) # marks it into current stack
            for adj in graph[v]:
                if hasCycle(adj, stack):
                    return(True)
            stack.pop()
            return(False)

        for course in graph:
            stack = [] # new stack every time
            if hasCycle(course, stack):
                return(False)
            
        return(True)
      
   # https://leetcode.com/problems/course-schedule/   

class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        
        lenList = 0
        cur = head
        while cur:
            lenList += 1
            cur = cur.next
        
        #if lenList == 1:
        #    return(head.val)
        for i in range(lenList//2):
            head = head.next
        return(head)
    
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        
        fast = slow = head
        
        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next 
        
        if fast.next:
            return(slow.next)
        return(slow)
    
    # https://leetcode.com/problems/reorder-list/

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        min_prev = max_prev = global_max = nums[0]
        
        for num in nums[1:]:
            curr_min = min(min_prev*num,max_prev*num,num)
            curr_max = max(min_prev*num,max_prev*num,num)
            global_max = max(global_max,curr_max)
            min_prev = curr_min
            max_prev = curr_max
            
        return(global_max)

class Solution:
    def splitArray(self, nums: List[int], m: int) -> int:
        
        left, right = max(nums), sum(nums)
        while left < right:
            mid = left + (right - left)//2 
            if self.isSumLargeEnough(nums, m, mid):
                right = mid
            else:
                left = mid + 1
        return(left)
    
    
    def isSumLargeEnough(self, nums, m, trySum):
        
        array_ind = 0
        total_sum = 0
        for i in range(len(nums)):
            
            total_sum += nums[i]
            
            if total_sum > trySum:
                if array_ind == m - 1:
                    return(False)
                total_sum = nums[i]
                array_ind += 1
        
        return(total_sum <= trySum)


# https://leetcode.com/problems/split-array-largest-sum/

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        
        subTree = ''.join(self.dfs_print(subRoot))
        mainTree = ''.join(self.dfs_print(root))
        return(subTree in mainTree)
        
    def dfs_print(self, root):
        res = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node:
                res += '<' + str(node.val) + '>'
                stack.append(node.left)
                stack.append(node.right)
            else:
                res.append('*')
        
        return(res)
    
 # recursive

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        
        def dfs(root, subRoot):
            if not root:
                return(False)
            if root.val == subRoot.val and CheckTrees(root, subRoot):
                return(True)

            return(dfs(root.left, subRoot) or dfs(root.right, subRoot))

        def CheckTrees(tree1, tree2):

            if not tree1 and not tree2:
                return(True)

            if tree1 and not tree2 or not tree1 and tree2:
                return(False)

            if tree1.val != tree2.val:
                return(False)

            return CheckTrees(tree1.left, tree2.left) and CheckTrees(tree1.right, tree2.right)
        
        if not subRoot:
            return(True)
        
        return(dfs(root, subRoot))
            
# https://leetcode.com/problems/subtree-of-another-tree/

class Solution:
    def reverseBits(self, n: int) -> int:
        
        mask = 1 << 31
        res = 0
        for i in range(32):
            res << 1
            if n&1:
                res|= mask
            n >>= 1
            mask >>= 1
        return(res)

Class Solution:
        def frequencySort(self, nums: List[int]) -> List[int]:
        #from Collections import Counter
        c = Counter(nums)
        return(sorted(c.elements(), key=lambda n:(c[n],-n)))

class Solution:
    def maxArea(self, height: List[int]) -> int:
        
        left,right = 0, len(height)-1
        water = -1
        while left < right:
            water = max(water,(right-left)*min(height[right],height[left]))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return(water)

class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        
        counts = collections.Counter()
        start, end = 0, 0 # window limits
        max_len = 0
        for end in range(len(s)):
            counts[s[end]] += 1
            if end - start + 1 > k + counts.most_common(1)[0][1]: # we want at most window length == k + most_common
                counts[s[start]] -= 1
                start += 1 # window shift
            max_len = max(max_len,end - start + 1)
        return(max_len)

class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        
        seen = {}
        for ind,val in enumerate(nums):
            if val in seen:
                ind_dist = seen[val]- ind
                if abs(ind_dist) <= k:
                    return(True)
            seen[val] = ind
        return(False)
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:

        """
        Do not return anything, modify head in-place instead.
        """
        
        # Divide list in two halves. Find half using fast and slow
        # Reverse second half
        # Merge the two halves
        
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        
        # the second list starts at slow.next
        
        sec_list = slow.next 
        slow.next = None # if we want to disconnect the two lists
        # now we will reverse sec_list
        curr = sec_list
        prev = None
        
        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt 
        sec_list = prev
        
        # now, if the list was 1->2->3  ->4->5, it will be  1->2->3  ->5->4
        # the two lists heads are: head and sec_list
        first_list = head
        
        while first_list and sec_list:
            first_list_nxt = first_list.next
            first_list.next = sec_list
            tmp = sec_list.next
            sec_list.next = first_list_nxt
            sec_list = tmp
            first_list = first_list_nxt
            
        











# https://leetcode.com/problems/reorder-list/

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        d = {}
        for num in nums:
            if num in d:
                d[num] = d[num] + 1
            else:
                d[num] = 1
        for key,val in d.items():
            if val == 1:
                return(key)

    def singleNumber(self, nums: List[int]) -> int:
        
        return(2*sum(set(nums))-sum(nums))

class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        
        latest_occ = {c:i for i,c in enumerate(s)}
        stack = ['!']
        visited = set()
        
        for ind,st in enumerate(s):
            if st in visited: continue # it's a duplicate dont need it
            
            while st < stack[-1] and latest_occ[stack[-1]] > ind:
                visited.remove(stack.pop())
            
            stack.append(st)
            visited.add(st)
        
        return("".join(stack[1:]))
        
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result = []
        def rec(x,y):
            if len(y)==1:
                result.append(x+y)
            else:
                for i,v in enumerate(y):
                    rec(x+[v],y[:i]+y[i+1:])
                        
        rec([],nums)
        return(result)
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        zero_rows = []
        zero_cols = []
        
        for ind_row,row in enumerate(matrix):
            for ind_col,col in enumerate(row):
                if not matrix[ind_row][ind_col]:
                    if ind_row not in zero_rows:
                        zero_rows.append(ind_row)
                    if ind_col not in zero_cols:
                        zero_cols.append(ind_col)
        
        for ind_row,row in enumerate(matrix):      
                for ind_col,col in enumerate(row):
                    if ind_col in zero_cols:
                        matrix[ind_row][ind_col] = 0                        
                    elif ind_row in zero_rows:
                        matrix[ind_row][ind_col] = 0
                        
# https://leetcode.com/problems/set-matrix-zeroes/

class Solution:
    def countPrimes(self, k:int) -> int:    
        if k < 3:
            return(0)
        isPrime = [True]*k
        isPrime[0] = isPrime[1] = False
        for i in range(2,int(k**0.5)+1):
            if isPrime[i]:
                for j in range(i*i,k,i): #e.g. for 7, start from 7*7=49
                    isPrime[j] = False
        return(sum(isPrime))
        

# Definition for singly-linked list.
#class ListNode:
#    def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
        
class Solution:
    def removeZeroSumSublists(self, head):
        node = ListNode(0)
        node.next = head
        head = node
        current_node = head
        d,dvalues,csum = {},[],0
        
        while current_node:
            csum += current_node.val
            if csum in d:
                d[csum].next = current_node.next
                while dvalues[-1]!= csum: # this is the first time csum was observed, havent saved csum yet
                    cur_last = dvalues.pop()
                    del d[cur_last]
            else:
                d[csum] = current_node
                dvalues.append(csum)
            
            current_node = current_node.next
        return(head.next)
            
class Solution:
    def countSubstrings(self, s: str) -> int:
        
        dp = [[False for i in range(len(s))] for j in range(len(s))]
        count = 0
        for i in range(len(s)):
            dp[i][i] = True
            count += 1
        
        
        for i in range(len(s)-1,-1,-1):
            for j in range(i+1, len(s)):
                if s[i] == s[j]:
                    if j-i==1 or dp[i+1][j-1]:
                        dp[i][j] = True
                        count += 1
        return(count)
      
 # https://leetcode.com/problems/palindromic-substrings/

# leetcode

class Solution(object):
    def getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
        if a == 0:
            return b
        elif b == 0:
            return a
        
        mask = (1 << 32) - 1

        # in Python, every integer is associated with its two's complement and its sign.
        # However, doing bit operation "& mask" loses the track of sign. 
        # Therefore, after the while loop, a is the two's complement of the final result as a 32-bit unsigned integer. 
        while b != 0:
            a, b = (a ^ b) & mask, ((a & b) << 1) & mask

        # a is negative if the first bit is 1
        if (a >> 31) & 1:
            return ~(a ^ mask)
        else:
            return a

class Solution:
    # @param s, a string
    # @return an integer
    def numDecodings(self, s):
        #dp[i] = dp[i-1] if s[i] != "0"
        #       +dp[i-2] if "09" < s[i-1:i+1] < "27"
        if s == "": return 0
        dp = [0]*(len(s)+1)
        dp[0] = 1
        for i in range(1, len(s)+1):
            if s[i-1] != "0":
                dp[i] += dp[i-1]
            if i != 1 and 9 < int(s[i-2:i]) < 27:  #"01"ways = 0
                dp[i] += dp[i-2]
        return dp[-1]

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        
        seen = {}
        for i,v in enumerate(nums):
            if (target - v) in seen:
                return[seen[target - v],i]
            seen[v] = i

class Solution:
    def reverse(self, x: int) -> int:
        if not x:
            return(x)
        sign = x/abs(x)
        x = abs(x)
        digits = []
        minv = -2**31
        maxv = 2**31-1
        while x != 0:
            digits.append(x%10)
            x = x // 10
        rev = 0
        for i,v in enumerate(digits):
            rev += v * 10**(len(digits)-i-1)
        rev = int(sign*rev)
        if minv <= rev <= maxv:
            return(rev)
        else:
            return(0)
            

class CustomStack:

    def __init__(self, maxSize: int):
        self.stack = []
        self.maxsize = maxSize



    def push(self, x: int) -> None:
        if len(self.stack) < self.maxsize:
            self.stack.append(x)
 
        

    def pop(self) -> int:
        if self.stack:
            return(self.stack.pop())
        else:
            return(-1)
        

    def increment(self, k: int, val: int) -> None:
        if len(self.stack) <= k:
            for elem in range(len(self.stack)):
                self.stack[elem] = self.stack[elem] + val
        else:
            for elem in range(k):
                self.stack[elem] = self.stack[elem] + val
        
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        
        dp = [0]*len(nums)
        dp[0] = nums[0]
        for i in range(1,len(nums)):
            dp[i] = max(dp[i-1] + nums[i],nums[i])
        return(max(dp))

class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        
        res = []
        def rec(x,y):
            if len(y)==1:
                if x+y not in res:
                    res.append(x+y)
            else:
                for i,v in enumerate(y):
                    rec(x+[v],y[:i] + y[i+1:])

        rec([],nums)    
        return(res)
        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.max_sum = float('-inf')
        self.dfs(root)
        
        return(self.max_sum)
    
    def dfs(self, root):
        
        if not root:
            return(0)
        
        left_sum = max(0,self.dfs(root.left))
        right_sum = max(0,self.dfs(root.right))
        
        self.max_sum = max(self.max_sum, left_sum + right_sum + root.val) #!!! this is if we dont consider any other path
        
        # but what we return is for further tree exploration, hence only one subtree 
        return(root.val + max(left_sum,right_sum))
      
# https://leetcode.com/problems/binary-tree-maximum-path-sum/

class Solution:
    def isValid(self, s: str) -> bool:
        parenth_stack = []
        corresp = {")":"(", "}":"{", "]":"["}
        for char in s:
            if char not in corresp:
                parenth_stack.append(char)
            else:
                if not parenth_stack:
                    return(False)
                if parenth_stack[-1] == corresp.get(char):
                    parenth_stack.pop()
                else:
                    return(False)
        return(parenth_stack == [])
      
# https://leetcode.com/problems/valid-parentheses/

class Solution:
    def findMin(self, nums: List[int]) -> int:
        left,right = 0, len(nums)-1
        while left < right:
            mid = (left + right)//2
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid
        return(nums[right])
        

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    ## Recursion
    #def searchBST(self, root: TreeNode, val: int) -> TreeNode:
    #    if root.val == val:
    #        return(root)
    #    elif root.val > val and root.left:
    #        return(self.searchBST(root.left,val))
    #    elif root.val < val and root.right:
    #        return(self.searchBST(root.right,val))
    #    else: 
    #        return(None)
        
    ## Binary search
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        l = root
        while l:
            if l.val == val:
                return(l)
            elif l.val > val:
                l = l.left
            else:
                l = l.right
        return(None)
        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# when we apply queue to that binary tree, we do BFS!
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        
        queue = collections.deque([(p,q)])
        while queue:
            p,q = queue.popleft()
            if not p and not q:
                continue
            if not (p and q):
                return(False)
            if p.val != q.val:
                return(False)
            queue.append((p.left, q.left))                     
            queue.append((p.right, q.right))
        
        return(True)
      
# when we apply stack, we do DFS !
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        
        stack = [(p,q)]
        while stack:
            p,q = stack.pop()
            if not p and not q:
                continue
            if not (p and q):
                return(False)
            if p.val != q.val:
                return(False)
            stack.append((p.left, q.left))                     
            stack.append((p.right, q.right))
        
        return(True)
      
      # https://leetcode.com/problems/same-tree/

class Solution:
    #def subsets(self, nums: List[int]) -> List[List[int]]:
    #    res = [[]]
    #    for num in nums:
    #        res += [curr + [num] for curr in res]
    #        
    #    return(res)
    
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def backtrack(first = 0, curr = []):
            # if the combination is done
            if len(curr) == k:  
                output.append(curr[:])
                return
            for i in range(first, n):
                # add nums[i] into the current combination
                curr.append(nums[i])
                # use next integers to complete the combination
                backtrack(i + 1, curr)
                # backtrack
                curr.pop()
        
        output = []
        n = len(nums)
        for k in range(n + 1):
            backtrack()
        return output
class Solution:
    def canArrange(self, arr: List[int], k: int) -> bool:
        
        from collections import Counter
        cnt = Counter([i%k for i in arr])
        for i in cnt:
            if i==0:
                if cnt[i]%2 != 0:
                    return(False)
            elif cnt[i] != cnt[k-i]:
                return(False)
        return(True)
        
class Solution:
    def subarraysDivByK(self, A: List[int], K: int) -> int:
        
        d = collections.defaultdict(int)
        count = 0
        d[0]=1
        ans = csum = 0
        for val in A:
            csum = (csum + val)%K
            ans += d[csum%K]
            d[csum] +=1          
        return(ans)
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        
        min_time = 0
        max_frequency = 0
        # make a dictionary with letter occurences!!
        
        #task_frequency={}
        #for i in range(len(tasks)):
        #    if task_frequency.get(tasks[i]) == None:
        #        task_frequency[tasks[i]] = 0
        #    task_frequency[tasks[i]] += 1
            #max_frequency = max(max_frequency, task_frequency[tasks[i]])
        
        task_frequency = Counter(tasks)
        
        max_frequency = task_frequency[max(task_frequency,key=task_frequency.get)]
        min_time = (max_frequency - 1)*n + max_frequency - 1
        
        for task in task_frequency.keys():
            if task_frequency[task] == max_frequency:
                min_time += 1   
        #tasks_with_maxfreq = task_frequency.values().count(max_frequency)
        #min_time += tasks_with_maxfreq
            
        return(max(len(tasks),min_time))  
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        num = head.val
        while head.next:
            num = (num * 2) + head.next.val
            head = head.next
        return num
class Solution:
    def countBits(self, n: int) -> List[int]:
        
        output = []
        for i in range(n+1):
            count = 0
            while i:
                i &=(i-1)
                count += 1
            output.append(count)
        return(output)
      
class Solution:
    def countBits(self, n: int) -> List[int]:
        
        count = [0]*(n+1)
        for i in range(n+1):
            count[i] = count[i>>1] + i%2
        return(count)

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        p = 1
        output = []
        for i,v in enumerate(nums):
            output.append(p)
            p *= v
        p = 1
        for i in range(len(nums)-1,-1,-1):
            output[i] *= p
            p*=nums[i]
            
        return(output)

class Solution:
    def romanToInt(self, s: str) -> int:
        
        if not s:
            return(s)
        
        dic = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
        num = 0
        for ind,st in enumerate(s):  
            if ind < len(s)-1:
                if st == 'I' and (s[ind+1] == 'V' or s[ind+1] == 'X'):
                    num -= 1
                elif st == 'X' and (s[ind+1] == 'L' or s[ind+1] == 'C'):
                    num -= 10
                elif st == 'C' and (s[ind+1] == 'D' or s[ind+1] == 'M'):
                    num -= 100
                else:
                    num += dic[st]
            else:
                num += dic[st]
        return(num)

        

class Solution:
    def isPalindrome(self, s: str) -> bool:
        new_st = ""
        for char in s:
            if char.isalnum():
                new_st += char.lower() 
        return(new_st =="".join(reversed(new_st)))
      
 # https://leetcode.com/problems/valid-palindrome/submissions/

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        
        rs = [amount + 1]*(amount + 1)
        rs[0] = 0
        
        for i in range(1,amount + 1):
            for c in coins:
                if i>=c:
                    rs[i] = min(rs[i],1+rs[i-c])
        if rs[amount] == amount + 1:
            return(-1)
        return(rs[amount])
             

class Solution:    
    def isValid(self, x, y, grid):
        m = len(grid)
        n = len(grid[0])
        if x < 0 or y < 0 or x >= m or y >= n:
            return False
        return True
    
    def numIslands(self, grid: List[List[str]]) -> int:
        
        if not grid or not grid[0]:
            return(0)
        # need to find number of DAGs
        
        self.directions = [[-1,0], [0,-1], [1,0], [0,1]]
        m = len(grid)
        n = len(grid[0])
        visited = [[False for _ in range(n)] for _ in range(m)]
        
        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1'  and (not visited[i][j]):
                    count += 1
                    self.dfs(grid, i, j, visited)
        return(count)
            
            
    def dfs(self, grid, i, j, visited):
        visited[i][j] = True
        for direc in self.directions:
            x, y = i + direc[0], j + direc[1]
            if self.isValid(x,y,grid) and grid[x][y] == '1' and (not visited[x][y]):
                self.dfs(grid, x, y, visited) 
# Instead of visited, we could instead change the grid[][] itself. Whenever we visit a node, we can do grid[i][j] = '*'
# and in the if condition of the recursion, ask grid[x][y] == '1'
class Solution:    
    def isValid(self, x, y, grid):
        m = len(grid)
        n = len(grid[0])
        if x < 0 or y < 0 or x >= m or y >= n or grid[x][y] != '1':
            return False
        return True
    
    def numIslands(self, grid: List[List[str]]) -> int:
        
        self.directions = [[-1,0], [0,-1], [0,1], [1,0]]
        
        m = len(grid)
        n = len(grid[0])
        
        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    self.dfs(grid, i, j)
                    count += 1
        return(count)
    
    def dfs(self, grid, i, j):
        grid[i][j] = '*'
        for direc in self.directions:
            x, y = i + direc[0], j + direc[1]
            if self.isValid(x, y, grid):
                self.dfs(grid, x, y)                
                
# https://leetcode.com/problems/number-of-islands/

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        
        dic, start, max_len = {}, 0, 0

        for ind,st in enumerate(s):
            if st in dic:
                max_len = max(ind - start,max_len)
                start = max(start,dic[st] + 1)

            dic[st] = ind
        
        return(max(max_len,len(s)-start))

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        
        dp = [1]*len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i] and dp[i] < dp[j] + 1:
                    dp[i] = dp[j] + 1
        return(max(dp))

class Solution:
    def hammingWeight(self, n: int) -> int:
        
        count = 0
        mask = 1
        for i in range(32):
            if n&mask:
                count += 1
            mask <<= 1
        return(count)
      
      
class Solution:
    def hammingWeight(self, n: int) -> int:
        
        count = 0
        while n:
            n = n&(n-1)
            count += 1
        return(count)

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        
        #if nums ==[] or len(nums)==1:
        #    return len(nums)
        #j = 1
        #for i in range(1,len(nums)):
        #    if nums[j] == nums[j-1]:
        #        del nums[j]
        #        j -= 1
        #    j += 1
        #return(len(nums))

        nums[:] = set(nums)
        nums.sort()
        return len(nums)
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        
        dp = [0]*(amount +1)
        dp[0] = 1
        for coin in coins:
            for i in range(1,amount+1):
                if i>= coin:
                    dp[i] += dp[i-coin]
        return(dp[amount])

    # Difference betweem coinChange2 (no duplicates) and combination sum (duplicates matter:order matters): In the comb sum for each position you try each coin. 
    # In that way, you are distinguishing two different orderings. 
    # If you instead loop through every position for each coin, then you are imposing a specific ordering

# with dp, @lru_cache is a cache to memoize recent calls. None ameans it can be inifnitely expanded

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        
        num_set = set(nums)
        
        @lru_cache(None)
        def dp(x):
            if x-1 in num_set:
                return(1+ dp(x-1))
            return(1)
        
        res = 0
        for num in num_set:
            res = max(dp(num), res)
        
        return(res)
      
# with hashSet 
# Although the time complexity appears to be quadratic due to the while loop nested within the for loop, closer inspection reveals it to be linear. Because the while loop is reached only when currentNum marks the beginning of a sequence (i.e. currentNum-1 is not present in nums), the while loop can only run for nn iterations throughout the entire runtime of the algorithm. This means that despite looking like O(n \cdot n)O(n⋅n) complexity, the nested loops actually run in O(n + n) = O(n)O(n+n)=O(n) time. 
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        
        num_set = set(nums)
        
        res = 0
        for num in num_set: # set lookup is 0(1)
            
            if num - 1 not in num_set: # i'm always looking for numbers greater than num, to avoid multipple lookups.
                current_num = num
                current_streak = 1
                
                while  current_num + 1 in num_set:
                    current_num += 1
                    current_streak += 1
            
                res = max(res, current_streak)
        
        return(res)
# https://leetcode.com/problems/longest-consecutive-sequence/

class Solution:
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        
        rows = [{} for i in range(9)]
        cols = [{} for i in range(9)]
        box = [{} for i in range(9)]
        
        for i in range(9):
            for j in range(9):
                num = board[i][j]
                if num!= '.':
                    num = int(num)
                    box_index = 3*(i//3) + j//3
                    
                    if rows[i].get(num, 0) ==0:
                        rows[i][num] = True
                    else:
                        return(False)
                    
                    if cols[j].get(num, 0) ==0:
                        cols[j][num] = True
                    else:
                        return(False)
                        
                    if box[box_index].get(num, 0) ==0:
                        box[box_index][num] = True
                    else:
                        return(False)
        return(True)
            
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        
        res = 0
        if not root:
            return(0)
        
        stack = [root]
        d = {}
        d[root] = 1
        while stack:
            curr = stack.pop()
            if curr.left:
                stack.append(curr.left)
                d[curr.left] = d[curr] + 1

            if curr.right:
                stack.append(curr.right)
                d[curr.right] = d[curr] + 1
        
        return(max(d.values()))
      
      # recursive 
      
      # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return(0)
        return(1 + max(self.maxDepth(root.left), self.maxDepth(root.right)))
      
  # https://leetcode.com/problems/maximum-depth-of-binary-tree/

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if not nums:
            return(nums)
        output = []
        nums1 = sorted(nums)
        for ind,target in enumerate(nums1):
            if ind > 0 and nums1[ind] == nums1[ind-1]:
                continue
            dic = {}
            for i,v in enumerate(nums1):
                if i != ind:
                    if v not in dic:
                        dic[-target - v] = i
                    else:
                        output.append([target,v,-target -v])
        out_set = set(tuple(sorted(x)) for x in output)
        output = list(out_set)
        return(output)
            

class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res, stack = [], [(root,False)]
        while stack:
            node, visited = stack.pop() #the last element
            if node:
                if visited:
                    res.append(node.val)
                else:
                    stack.append((node.right, False))
                    stack.append((node, True))
                    stack.append((node.left, False))
        return res        

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return(root)
        level = [root]
        res = []
        while level:
            level_res = []
            queue = []
            for node in level:
                level_res.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level_res)
            level = queue
        return(res)

# https://leetcode.com/problems/binary-tree-level-order-traversal/

class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        if nums == []:
            return(0)
        for i in range(len(nums)):
            if nums[i] == target:
                return(i)
            elif nums[i] > target:
                return(i)
        return(i+1)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return('')
        queue = collections.deque([root])
        
        res = []
        while queue:
            node = queue.popleft()
            if not node:
                res.append('*')
            else:
                res.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
        return(','.join(res))
        
        
    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if data == '':
            return([])
        to_be_filled = data.split(',')
        
        root = TreeNode(int(to_be_filled[0]))
        queue = collections.deque([root])
        
        i = 1
        while queue:
            node = queue.popleft()
            if to_be_filled[i] != '*':
                node.left = TreeNode(int(to_be_filled[i]))
                queue.append(node.left)
            if to_be_filled[i+1] != '*':
                node.right = TreeNode(int(to_be_filled[i+1]))
                queue.append(node.right)
            i += 2
        return(root)
            
        
        

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))

# https://leetcode.com/problems/serialize-and-deserialize-binary-tree/

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

# Similar to given an array of positivie numbers, find the maximum sub of non-adjacent numbers
class Solution:
    def rob(self, nums: List[int]) -> int:
        
        cache = [0]*len(nums)
        cache[0] = nums[0]
        if len(nums)<=2:
            return(max(nums))
        cache[1] = max(nums[0],nums[1])
                   
        for i in range(2,len(nums)):
            cache[i] = max(nums[i] + cache[i-2],cache[i-1])
        
        return(cache[-1])
                   

class Solution:
    def rob(self, nums: List[int]) -> int:
        # Break the circle by considering array[0..n-1], [1..n]
        if len(nums)<=2:
            return(max(nums))
        
        def rob_house(left_ind,right_ind):
            
            numss = nums[left_ind:right_ind]
            cache = [0]*(len(numss))
            cache[0] = numss[0]
            cache[1] = max(numss[0],numss[1])
            for i in range(2,len(numss)):
                cache[i] = max(numss[i] + cache[i-2],cache[i-1])

            return(cache[-1])
        
        return(max(rob_house(0,len(nums)-1),rob_house(1,len(nums))))           

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

# Overall approach is graph traversal. The trick is in cloning.

# Use a dictionary with node as key and the clone as value
# when visiting the neighbors, create a key-value pair for each child node.
# add that key value pair to the neighbor for the parent node if that neighboring node has not been visited.
# Ensuring that the neighboring node is not visited is crucial as it could lead to a spiral (i made that mistake)
# return the value of the node being passed in


"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""
# DFS recursively
class Solution:
    
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return
        dic = {}
        node_clone = Node(node.val, [])
        dic = {node: node_clone}
        self.dfs(node, dic)
        return(node_clone)
    
    def dfs(self, node, dic):
        for neigh in node.neighbors:
            if neigh not in dic:
                neighCopy = Node(neigh.val, [])
                dic[neigh] = neighCopy
                dic[node].neighbors.append(neighCopy)
                self.dfs(neigh, dic)
            else:
                dic[node].neighbors.append(dic[neigh])
                
                
    # DFS iterative
class Solution:
    
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return
        nodeCopy = Node(node.val, [])
        dic = {node: nodeCopy}
        stack = [node]
        while queue:
            curr = stack.pop()
            for neigh in curr.neighbors: 
                if neigh not in dic:
                    neighCopy = Node(neigh.val, [])
                    dic[neigh] = neighCopy
                    dic[curr].neighbors.append(neighCopy)
                    stack.append(neigh)
                else:
                    dic[curr].neighbors.append(dic[neigh])
        return(nodeCopy)
    
# BFS same as DFS iterative, but queue instead of stack
    class Solution:
    
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return
        nodeCopy = Node(node.val, [])
        dic = {node: nodeCopy}
        queue = collections.deque([node])
        while queue:
            curr = queue.popleft()
            for neigh in curr.neighbors: 
                if neigh not in dic:
                    neighCopy = Node(neigh.val, [])
                    dic[neigh] = neighCopy
                    dic[curr].neighbors.append(neighCopy)
                    queue.append(neigh)
                else:
                    dic[curr].neighbors.append(dic[neigh])
        return(nodeCopy)
    
                

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result = []
        def rec(x,y):
            if len(y)==1:
                result.append(x+y)
            else:
                for i,v in enumerate(y):
                    rec(x+[v],y[:i]+y[i+1:])
                        
        rec([],nums)
        return(result)
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        
        left, right = 0, len(nums) - 1
        
        while left <= right:
            
            pivot = left + (right-left)//2
            if nums[pivot] == target:
                return(pivot)
            if nums[pivot] <= target:
                left = pivot + 1
            else:
                right = pivot - 1
        return(-1)

# https://leetcode.com/problems/binary-search/

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)-1
        
        
        while left <= right:
            mid = (left + right) >> 1
            if target == nums[mid]:
                return(mid)
            if nums[mid] >= nums[left]: #right-rotated
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else: #left-rotated
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return(-1)

class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return(sorted(t)==sorted(s)) 
        
        #d = collections.defaultdict(int)
        #if len(t) == len(s):
        #    for i in range(len(t)):
        #        d[ord(t[i])-ord('a')] +=1
        #        d[ord(s[i])-ord('a')] -=1
        #    return(list(d.values())==[0]*len(d.values()))
        #else:
        #    return(False)
        
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
