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