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

k_neighbour=4


datanames = ['upd']
data_dir_path = './dataset/'
result_dir_path = './Results/'
if not os.path.exists(result_dir_path):
    os.makedirs(result_dir_path)

result=[] #empty list to store results in each loop of data 


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
                        
            ## distance sum between k nearest agents in a coalition and a specific agents
            def neighbourhood_distance(agent_i, coalition_j, task_idx):
                for a in range(len(coalition_j)):
                    agent_dist=(dist_agent_agent[agent_i])[coalition_j]
                    idx_near_agent=agent_dist.argsort()[:k_neighbour].tolist()
                    #print(idx_near_agent)
                    near_agent=[coalition_j[t] for t in idx_near_agent]
                    dist_near_agent=(dist_agent_task[:,task_idx])[near_agent]
                    return sum(dist_near_agent)
            
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
            
            
            ################# DS-SCSGA-MF
            ####Phase-1
            #########
            #3.Distance and Task Satisfaction based Assignment Algo (DTA)
            def dta_algo():
                initial_solution = [[] for i in range(m_tasks)]
                
                for a in range(n_agents):
                    dist1 = dist_agent_task[a]    #take distances between a_i and all t_j
                    sort_idx = np.argsort(dist1)  #sort t_j or, C_j of min distances
                    for t in range(m_tasks):
                        task_col = initial_solution[sort_idx[t]]   #select the assigned agents in C_j or, t_j 
                        task_col_value = sum(agents[task_col])     #calculate value of C_j
                        task_val = tasks[sort_idx[t]]              #values of t_j
                        satisfy_val = (task_col_value - task_val)  #(value of C_j) - (value of t_j)
                        
                        if (all(x >= 0 for x in satisfy_val)):     # check if all values are >=0
                            continue                               #t_j satisfied, no need of agent assignment
                        else:
                            initial_solution[sort_idx[t]].append(a) #t_j not satisfied, assign a_i
                            break
                
                #find the remaining agents left to be assigned
                rem_agents = [ele for ele in list(np.arange(n_agents)) if ele not in sum(initial_solution, [])]
                dist2 = dist_agent_task[rem_agents]  #take agent-task disances for remaining agents
                #assign rem_agents in minimum distanced task
                [initial_solution[np.argmin(dist2[r])].append(rem_agents[r]) for r in range(len(rem_agents))]
                        
                initial_count = task_satisfy_count(initial_solution)
                return initial_solution, initial_count
            
            
            ####Phase-2
            def ds_scsga(initial_solution, initial_count):
                T_t=copy.deepcopy(initial_solution)  #copy of initial solution
                
                for i in range(n_agents):
                    agent_idx=[[k, item.index(i)] for k, item in enumerate(T_t) if i in item] #finding index of a_i in C_j
                    agent_idx=agent_idx[0][0]  #taking index of a_i in C_j
                    T_t[agent_idx].remove(i)   #initially remove a_i from that C_j
                    
                    sat_count=[]
                    for j in range(m_tasks):
                        T_t[j].append(i)                    #insert a_i in a C_j
                        curr_count=task_satisfy_count(T_t)  #count the number of tasks satisfied
                        sat_count.append(curr_count)        #save the count
                        T_t[j].remove(i)                    #remove the a_i from that C_j
                    sat_count=[0 if ele < initial_count else ele for ele in sat_count] #padding the C_j with 0 that has lower count than initial
                    order_task= np.argsort(sat_count)[::-1]  # Order tasks by max satisfy count
                    
                    #remove the task or C_j that has lower satisfy than initial 
                    if (sat_count.count(0) > 0 and sat_count.count(0) < m_tasks):
                        order_task = order_task[:-(sat_count.count(0))] 
                    else:
                        order_task = order_task
                    #neighbourhood distance sum
                    dist_sum=[]
                    for j in range(len(order_task)):
                        curr_dist_sum=neighbourhood_distance(i, T_t[order_task[j]], order_task[j])
                        dist_sum.append(curr_dist_sum)
                        
                    dist_sum = [d for d in dist_sum if d is not None]
                    if len(dist_sum)==0:
                        dist_sum.append(0)
                    final_assign_idx=order_task[np.argmin(dist_sum)]  # the task which has min dist sum
                    T_t[final_assign_idx].append(i)
                    
                    
                ########### Final Results
                ds_sol = T_t
                ds_sol_val = solution_value(T_t)
                ds_sol_count = task_satisfy_count(ds_sol)
                
                return ds_sol, ds_sol_val, ds_sol_count
            
            ############ Functions Run for Final Results ######### 
            #print(var)
            ### 3. DTA + DS_SCSGA 
            dta_start = time.time()
            sol3, count3 = dta_algo()
            dta_ds_sol, dta_ds_val, dta_ds_count = ds_scsga(sol3, count3)
            dta_end = time.time()
            dta_ds_time = (dta_end - dta_start)
            
            ################# ALL FINAL RESULTS PRINT ###################
            rows_name = [datanames[t]]
            result_data = {
                'Value' : [dta_ds_val],
                'Count' : [dta_ds_count],
                'Time' : [dta_ds_time]
            }
            result.append(pd.DataFrame(result_data, index = rows_name))

                        
            
    print(datanames[t],"Completed")
    
    result=pd.concat(result)
    result.to_csv(result_dir_path+datanames[t]+'_can.csv')
    result.to_csv(result_dir_path+'can.csv', mode='a')
    result=[]
