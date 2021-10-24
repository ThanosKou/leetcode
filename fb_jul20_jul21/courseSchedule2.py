class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        
        if not prerequisites:
            return([i for i in range(numCourses)])
        
        graph = {i:set() for i in range(numCourses)}
        in_degree = [0]*numCourses
        
        for course_pair in prerequisites:
            nxt, prereq = course_pair
            graph[prereq].add(nxt)
            in_degree[nxt] += 1
        
        queue = collections.deque([])
        for cor in range(len(in_degree)):
            if in_degree[cor] == 0:
                queue.append(cor)
        
        visited = []
        while queue:
            cor = queue.popleft()
            visited.append(cor)
            
            for adj in graph[cor]:
                in_degree[adj] -= 1
                if in_degree[adj] == 0:
                    queue.append(adj)
        
        if len(visited) == numCourses:
            return(visited)
        else:
            return([])
            



# https://leetcode.com/problems/course-schedule-ii/
