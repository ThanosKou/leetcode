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
