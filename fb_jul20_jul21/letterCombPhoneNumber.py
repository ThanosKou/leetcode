class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        
        dic = {"2":"abc",
               "3":"def",
               "4":"ghi",
               "5":"jkl",
               "6":"mno",
               "7":"pqrs",
               "8":"tuv",
               "9":"wxyz"}
        
        if digits:
            all_combinations = ['']
        else:
            all_combinations = []
        
        
        for digit in digits:
            cur_combinations = list()
            for letter in dic[digit]:
                for entry in all_combinations:
                    cur_combinations.append(entry + letter)
            all_combinations = cur_combinations
        return(all_combinations)

#
