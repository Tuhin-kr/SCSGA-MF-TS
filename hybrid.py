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
misc_file = open('hybrid_misc_result.txt', 'w')



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
                        
            ### Distance between a single coalition and a task          
            def coalition_task_dist(coalition_list, task_id):
                if len(coalition_list)==0:
                    a_t_dist=cdist((np.zeros(1,features)), tasks[task_id], metric='euclidean')
                    return sum(a_t_dist)
                else:
                    a_t_dist=(dist_agent_task[coalition_list])[:,task_id]
                    return sum(a_t_dist)
            
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
            
            
            ### 3. Hill Climb Approach
            def hybrid_hill_climb():
                hill_temp_start = time.time()   #start with per agent time                
                #phase-1: initial solution generation by Greedy
                agent_permut = random.sample(list(np.arange(n_agents)), n_agents) #take random permutation of all agents
                
                greedy_sol = [[] for i in range(m_tasks)]    #start with m blank coalitions
                for i in range(n_agents):
                    temp_assign=[]
                    for j in range(m_tasks):
                        greedy_sol[j].append(agent_permut[i])              #initially assign a_i in t_j
                        temp_assign.append(solution_value(greedy_sol) )    # store the solution value
                        greedy_sol[j].remove(agent_permut[i])              #remove a_i from t_j
                    greedy_sol[np.argmin(temp_assign)].append(agent_permut[i]) #finally assign a_i in minimum valued t_j
                

                #phase-2: improve initial solution
                initial_solution=copy.deepcopy(greedy_sol)                   #assign random solution as initial solution
                agent_permut = random.sample(list(np.arange(n_agents)), n_agents) #random all agents permutation
                
                running_time = 0          #time flag to save time
                success_flag = False
                #while success==True and budget<hc_budget:                      #define user-budget in place of 10
                
                while not success_flag:
                    
                    success_flag = True
                    
                    for a in range(n_agents):
                        cur_agent = agent_permut[a]                      #select the current agent a_i
                        #finding index of a_i in C_j
                        agent_idx=[[k, item.index(cur_agent)] for k, item in enumerate(initial_solution) if cur_agent in item] 
                        agent_idx=agent_idx[0][0]                       #taking index of a_i in C_j
                        pry_c_t_val=coalition_task_dist(initial_solution[agent_idx], agent_idx) #store the primary distance coalition U a_i & task
                        initial_solution[agent_idx].remove(cur_agent)   #initially remove a_i from that C_j
                        
                        all_c_t_val=[]
                        for t in range(m_tasks):
                            initial_solution[t].append(cur_agent)               #sequentially add a_i in C_j
                            c_t_val=coalition_task_dist(initial_solution[t], t) #calculate coalition value with a_i
                            all_c_t_val.append(c_t_val)                        #store the coalition value
                            initial_solution[t].remove(cur_agent)             #initially remove a_i from C_j
                        #check if placing a_i in any C_j generate min value than previous assignment of a_i    
                        if min(all_c_t_val) < pry_c_t_val:            
                            initial_solution[np.argmin(all_c_t_val)].append(cur_agent) #put a_i in lowest valued C_j
                            #success_obtain.append(1)              #if any improvement obtained just assign 1
                            success_flag = False                # if any improvements are possible
                        else:
                            initial_solution[agent_idx].append(cur_agent)  #remain a_i in same C_j
                        
                        #### Checking of time budget
                        if ((a+1) % 100 == 0):
                            hybrid_time2 = time.time()  
                            running_time += (hybrid_time2 - hill_temp_start)  #per agent time
                            hill_temp_start = hybrid_time2
                            if (running_time > time_budget):
                                success_flag = True                        #if reached stop outer while loop 
                                #print("Exceed Time=",time_flag, file=misc_file)
                                break                                #also break from agent loop    
                        
                    #if len(success_obtain)==0:  #NO success otained 
                    #    success=False
                    #else:
                    #    success=True            #success obtained
                    
                    
                    #budget+=1                                                #increase budget
                
                
                ########### Final Results
                hybrid_sol = initial_solution
                hybrid_sol_val = solution_value(hybrid_sol)
                hybrid_sol_count = task_satisfy_count(hybrid_sol)
                
                return hybrid_sol, hybrid_sol_val, hybrid_sol_count

            ## Function Run for Final Results 
            ### Hill Climb for SCSGA-MF 
            hybrid_start = time.time()
            hybrid_sol, hybrid_val, hybrid_count = hybrid_hill_climb()
            hybrid_time = (time.time() - hybrid_start)
            
            
            
            ################# ALL FINAL RESULTS PRINT ###################
            rows_name = [datanames[t]]
            result_data = {
                'Value' : [hybrid_val],
                'Count' : [hybrid_count],
                'Time' :  [hybrid_time],
                'Data inst': [inst],
                'Variane'  : [var]
            }
            result.append(pd.DataFrame(result_data, index = rows_name))
            temp_result = pd.DataFrame(result_data, index = rows_name)
            temp_result.to_csv(result_dir_path+datanames[t]+'_hybrid.csv', header=False, mode='a')

                             
    print(datanames[t],"Completed")
    
    result=pd.concat(result)
    #result.to_csv(result_dir_path+datanames[t]+'_hybrid.csv')
    result.to_csv(result_dir_path+'hybrid.csv', mode='a')
    result=[]



misc_file.close()




