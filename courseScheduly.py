# 1) DFS topo sort

class Solution:
        
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        
        if not prerequisites:
            return(True)
        
        graph = {i: set() for i in range(numCourses)}
        
        for course_pair in prerequisites:
            next_course, prereq = course_pair
            graph[next_course].add(prereq) # doesnt matter where to put next and prereq
        
        state = [0]*numCourses
        
        def hasCycle(v):
            if state[v] == -1:
                return(True)
            if state[v] == 1:
                return(False)
            state[v] = -1
            for adj in graph[v]:
                if hasCycle(adj):
                    return(True)
            state[v] = 1
            return(False)

        for course in graph:
            if hasCycle(course):
                return(False)       
        return(True)
      
      # 2) BFS topo sort , kahn's algo
        
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        
        if not prerequisites:
            return(True)
        
        
        graph = {i: set() for i in range(numCourses)}
        in_degree = [0]*numCourses
        
        for course_pair in prerequisites: # create graph
            next_course, prereq_course = course_pair
            graph[next_course].add(prereq_course)
            in_degree[prereq_course] += 1
        
        queue = collections.deque()
        for i in range(len(graph)):
            if in_degree[i] == 0:
                queue.append(i)
                
        visited = set()
        
        while queue:
            curr = queue.popleft()
            visited.add(curr)
            for adj in graph[curr]:
                in_degree[adj] -= 1
                if in_degree[adj] == 0:
                    queue.append(adj)
          
        return(len(visited) == numCourses)

    # 3) DFS with stack 
        
class Solution:
        
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        
        if not prerequisites:
            return(True)
        
        graph = {i: set() for i in range(numCourses)}
        
        for course_pair in prerequisites:
            next_course, prereq = course_pair
            graph[prereq].add(next_course)
        

        visited = set()
        
            
        def hasCycle(v, stack):
            if v in visited:
                if v in stack: # it's like state == -1
                    return(True)
                return(False) # its like state == 1
            visited.add(v)
            stack.append(v) # marks it into current stack
            for adj in graph[v]:
                if hasCycle(adj, stack):
                    return(True)
            stack.pop()
            return(False)

        for course in graph:
            stack = [] # new stack every time
            if hasCycle(course, stack):
                return(False)
            
        return(True)
      
   # https://leetcode.com/problems/course-schedule/   
