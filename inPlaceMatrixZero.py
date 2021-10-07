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
