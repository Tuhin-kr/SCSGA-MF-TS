import copy
import time
import os
import csv
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
            ## 1. LSA
            def linear_sum_assignment():
                unfulfilled_tasks = np.arange(m_tasks)
                free_agents = np.arange(n_agents)
                max_rounds = int(np.ceil(n_agents / m_tasks))
                assign_matrix = np.zeros((m_tasks, n_agents), dtype='int')
                
                for i in range(max_rounds):
                    row_assign, col_assign = lsa(dist_task_agent[unfulfilled_tasks][:,free_agents]) #LSA on task-agent dist
                    #print(i, unfulfilled_tasks[row_assign], free_agents[col_assign])
                    assign_matrix[unfulfilled_tasks[row_assign], free_agents[col_assign]] = 1 
                    delete_tasks = []
                    for tj in range(unfulfilled_tasks.shape[0]):
                        task_idx = unfulfilled_tasks[tj]
                        if ((tasks[task_idx] - agents[np.where(assign_matrix[task_idx])[0]].sum(axis=0)) > 0).sum() == 0:
                            delete_tasks.append(tj)
                    unfulfilled_tasks = np.delete(unfulfilled_tasks, np.array(delete_tasks, dtype=int))
                    free_agents = np.delete(free_agents, col_assign)
                
                # Assign free agents
                while len(free_agents) > 0:
                    row_assign, col_assign = lsa(dist_task_agent[:,free_agents])  #LSA on free agents
                    assign_matrix[row_assign, free_agents[col_assign]] = 1
                    free_agents = np.delete(free_agents, col_assign)
                
                # Final Coalition Structure Generation
                lsa_sol = [[] for i in range (m_tasks)]
                for i in range (m_tasks):
                    for j in range (len(assign_matrix[i])):
                        if assign_matrix[i][j]==1:
                            lsa_sol[i].append(j)
                            
                ########### Final Results
                lsa_sol = lsa_sol
                lsa_sol_val = solution_value(lsa_sol)
                lsa_sol_count = task_satisfy_count(lsa_sol)
                
                return lsa_sol, lsa_sol_val, lsa_sol_count

            ############ Function Run for Final Results ######### 
            ### LSA for SCSGA-MF 
            lsa_start = time.time()
            lsa_sol, lsa_val, lsa_count = linear_sum_assignment()
            lsa_end = time.time()
            lsa_time = (lsa_end - lsa_start)
            
            ################# ALL FINAL RESULTS PRINT ###################
            rows_name = [datanames[t]]
            result_data = {
                'Value' : [lsa_val],
                'Count' : [lsa_count],
                'Time' : [lsa_time]
            }
            result.append(pd.DataFrame(result_data, index = rows_name))
                      
            
    print(datanames[t],"Completed")
    
    result=pd.concat(result)
    result.to_csv(result_dir_path+datanames[t]+'_mdm.csv')
    result.to_csv(result_dir_path+'mdm.csv', mode='a')
    result=[]
