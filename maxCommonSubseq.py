
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
