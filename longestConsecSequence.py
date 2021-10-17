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
