class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        
        min_time = 0
        max_frequency = 0
        # make a dictionary with letter occurences!!
        
        #task_frequency={}
        #for i in range(len(tasks)):
        #    if task_frequency.get(tasks[i]) == None:
        #        task_frequency[tasks[i]] = 0
        #    task_frequency[tasks[i]] += 1
            #max_frequency = max(max_frequency, task_frequency[tasks[i]])
        
        task_frequency = Counter(tasks)
        
        max_frequency = task_frequency[max(task_frequency,key=task_frequency.get)]
        min_time = (max_frequency - 1)*n + max_frequency - 1
        
        for task in task_frequency.keys():
            if task_frequency[task] == max_frequency:
                min_time += 1   
        #tasks_with_maxfreq = task_frequency.values().count(max_frequency)
        #min_time += tasks_with_maxfreq
            
        return(max(len(tasks),min_time))  