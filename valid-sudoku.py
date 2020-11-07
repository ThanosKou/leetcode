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
            