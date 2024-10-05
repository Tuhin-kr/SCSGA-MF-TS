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
            #2.Probabilistic Distance based Assignment Algo (PDA)
            def pda_algo():
                initial_solution = [[] for i in range(m_tasks)]
                max_size = int(np.ceil(n_agents/m_tasks))
                epsilon = 1e-9
                
                for a in range(n_agents):
                    distances = dist_agent_task[a]    #take distances between a_i and all t_j
                    dist_eps = distances + epsilon    #add epsilon with each distance
                    inv_dist = 1/dist_eps             #inverse of each distance
                
                    col_size_j = [len(initial_solution[j]) for j in range(m_tasks)] #size of current coalitions
                    delta_j = max_size/ (np.array(col_size_j) + 1)        #calculate delta value for each coalition
                    
                    prob_numerator = (inv_dist * delta_j)             #value of numerator
                    prob_init = prob_numerator / sum(prob_numerator)  #initial probability
                    prob_final = np.cumsum(prob_init)                #final probability (cum sum of initial prob.)
                    
                    rand_no = np.random.rand()           #select a random no.
                    assign_idx = [list(prob_final).index(i) for i in prob_final if i>rand_no] #find the range of random no. in final_probability
                    initial_solution[assign_idx[0]].append(a)  #assign a_i in that range
                    
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
            ### 2. PDA + DS_SCSGA 
            pda_start = time.time()
            sol2, count2 = pda_algo()
            pda_ds_sol, pda_ds_val, pda_ds_count = ds_scsga(sol2, count2)
            pda_end = time.time()
            pda_ds_time = (pda_end - pda_start)
            #print(var)
            
            ################# ALL FINAL RESULTS PRINT ###################
            rows_name = [datanames[t]]
            result_data = {
                'Value' : [pda_ds_val],
                'Count' : [pda_ds_count],
                'Time' : [pda_ds_time]
            }
            result.append(pd.DataFrame(result_data, index = rows_name))

                        
            
    print(datanames[t],"Completed")
    
    result=pd.concat(result)
    result.to_csv(result_dir_path+datanames[t]+'_pbics.csv')
    result.to_csv(result_dir_path+'pbics.csv', mode='a')
    result=[]
