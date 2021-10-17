# 1) DFS topo sort

class Solution:
        
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        
        if not prerequisites:
            return(True)
        
        def addEdge(graph, u, v):
            graph[u].append(v)
        
        graph = [[] for _ in range(numCourses)]
        visit = [0 for _ in range(numCourses)]
        
        for course_pair in prerequisites: # create graph
            next_course, prereq_course = course_pair
            graph[next_course].append(prereq_course)
            
        for i in range(numCourses):
            if not self.dfs(graph, visit, i):
                return(False)
        return(True)
        # we need to check if there is a cycle, and also check for number of nodes visited --> DFS?
        
    def dfs(self, graph, visit, i):
        # -1 means currently being visited
        # 1 was visited previosuly
        if visit[i] == -1:
            return(False) # cycle
        if visit[i] == 1:
            return(True) 
        visit[i] = -1
        for j in graph[i]:
            if not self.dfs(graph, visit, j):
                return(False)
        visit[i] = 1
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

      
      
   # https://leetcode.com/problems/course-schedule/   
