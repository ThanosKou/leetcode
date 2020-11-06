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