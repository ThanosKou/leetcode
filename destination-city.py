class Solution:
    def destCity(self, paths: List[List[str]]) -> str:
        
        from collections import Counter
        c_arriving = Counter([city[1] for city in paths])
        c_leaving = Counter([city[0] for city in paths])
        for city in c_arriving:
            if city in c_arriving and city not in c_leaving:
                return(city)
            
            