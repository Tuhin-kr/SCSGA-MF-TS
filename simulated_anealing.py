import copy
import time
import os
import random
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment as lsa


#reading the values of input varibale
confile = open('input_variable.txt', 'r')
con=[]
for line in confile:
    name, value = line.split("=")
    value = value.strip()
    con.append(int(value))
n_agents=con[0]
m_tasks=con[1]
features=con[2]
distribution_instances=con[3] 
confile.close()  


datanames = ['upd']
data_dir_path = './dataset/'
result_dir_path = './Results/'
if not os.path.exists(result_dir_path):
    os.makedirs(result_dir_path)

result=[] #empty list to store results in each loop of data 
time_budget = 3600 # Change as per need
misc_file = open('sim_misc_result.txt', 'w')



for t in range(len(datanames)):
    for inst in range(distribution_instances):
        for var in range(5):
    
            main_data=np.load(data_dir_path+datanames[t]+str(inst)+'_v'+str(var)+'.npy') 
            
            agents = main_data[0:n_agents] # from row 0:n_agents
            tasks = main_data[n_agents:]  #from row n_agents:
            
            
            
            
            
            ### starting of method
            
            ########## Required Functions
            dist_agent_task = cdist(agents, tasks, metric='euclidean')
            dist_task_agent = (dist_agent_task).T
            dist_agent_agent = cdist(agents, agents, metric='euclidean')
            
            ### Task Satisfy Count
            def task_satisfy_count(col_struct):
                count=0
                for j in range(len(col_struct)):
                    skill_dim = sum(agents[col_struct[j]])- tasks[j] # v(C_j) - s(t_j)
                    #print(skill_dim)
                    if (all(x >= 0 for x in skill_dim)):  # check if all values are >=0
                        count+=1                          # task satisfied and increase count
                    else:
                        count=count
                return count
                        
                        
            #### Solution value            
            def solution_value(coalition_structure):
                all_coalition_value=[]
                for k in range (len(coalition_structure)):
                    if len(coalition_structure)==0:
                        all_coalition_value.append(0)
                    else:
                        temp1=(dist_agent_task[:,k])[coalition_structure[k]]
                        temp2=sum(temp1)
                        all_coalition_value.append(temp2)
                return sum(all_coalition_value)
            
            
            #### 4. Simulated Annealing Approach 
            def simulated_anealing():
                sim_ann_start = time.time()   #start with per agent time                
                #phase-1: initial solution generation
                random_solution=[[] for i in range(m_tasks)]    #start with m blank coalitios
                max_size = int(np.ceil(n_agents/m_tasks))      #maximum size of each coalition (balanced)
                for a in range(n_agents):
                    random_task=random.sample(list(np.arange(m_tasks)), m_tasks) #random indices of m tasks
                    for t in range(m_tasks):
                        curr_size = len(random_solution[random_task[t]])   #check the current size of C_j
                        if (curr_size  < max_size):
                            random_solution[random_task[t]].append(a)  #if C_j is not full then put a_i
                            break
                        else:
                            continue
                
                #phase-2: improve initial solution
                initial_solution=copy.deepcopy(random_solution)     #assign random solution as initial solution
                
                runnin_time = 0          #time flag to save time

                while runnin_time < time_budget:
                    cur_agent = np.random.randint(0,n_agents)      #select a random agent a_i
                    cur_task =  np.random.randint(0,m_tasks)       #select a random task t_j
                    
                    #finding the current index of a_i in C_j
                    cur_agent_idx=[[k, item.index(cur_agent)] for k, 
                                   item in enumerate(initial_solution) if cur_agent in item] 
                    cur_agent_idx=cur_agent_idx[0][0]                 #taking index of a_i in C_j
                    pry_sol_val=solution_value(initial_solution)      #calculate solution value with a_i in C_j
                    initial_solution[cur_agent_idx].remove(cur_agent) #initially remove a_i from that C_j
                    
                    initial_solution[cur_task].append(cur_agent)   #put a_i in random task t_j
                    new_sol_val=solution_value(initial_solution)   #calculate solution value with a_i in C_j
                    initial_solution[cur_task].remove(cur_agent)   #initially remove a_i from that C_j
                    
                    if new_sol_val < pry_sol_val:             #check improvement is achieved or not
                        initial_solution[cur_task].append(cur_agent) #if YES, put a_i in random task t_j
                    else:
                        initial_solution[cur_agent_idx].append(cur_agent)  #if NO, retain a_i in previous C_j     
                    
                    
                    #### Checking of time budget
                    sim_ann_start2 = time.time()
                    runnin_time += sim_ann_start2 - sim_ann_start  #per agent time
                    sim_ann_start = sim_ann_start2
                    
                    #budget+=1 
                ########### Final Results
                sim_sol = initial_solution
                sim_sol_val = solution_value(sim_sol)
                sim_sol_count = task_satisfy_count(sim_sol)
                
                return sim_sol, sim_sol_val, sim_sol_count

            ## Function Run for Final Results 
            ### Hill Climb for SCSGA-MF 
            sim_start = time.time()
            sim_sol, sim_val, sim_count = simulated_anealing()
            sim_time = (time.time() - sim_start)
            
            
            
            ################# ALL FINAL RESULTS PRINT ###################
            rows_name = [datanames[t]]
            result_data = {
                'Value' : [sim_val],
                'Count' : [sim_count],
                'Time' : [sim_time],
                'Data inst': [inst],
                'Variane'  : [var]
            }
            result.append(pd.DataFrame(result_data, index = rows_name))
            temp_result = pd.DataFrame(result_data, index = rows_name)
            temp_result.to_csv(result_dir_path+datanames[t]+'_simulated.csv', header=False, mode='a')

                        
            
    print(datanames[t],"Completed")
    
    result=pd.concat(result)
    #result.to_csv(result_dir_path+datanames[t]+'_simulated.csv')
    result.to_csv(result_dir_path+'simulated.csv', mode='a')
    result=[]


misc_file.close()



