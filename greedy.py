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
            
            
            ### 2. Greedy Algorithm
            def greedy():
                agent_permut = random.sample(list(np.arange(n_agents)), n_agents) #take random permutation of all agents
                
                greedy_sol = [[] for i in range(m_tasks)]    #start with m blank coalitions
                for i in range(n_agents):
                    temp_assign=[]
                    for j in range(m_tasks):
                        greedy_sol[j].append(agent_permut[i])              #initially assign a_i in t_j
                        temp_assign.append(solution_value(greedy_sol) )    # store the solution value
                        greedy_sol[j].remove(agent_permut[i])              #remove a_i from t_j
                    greedy_sol[np.argmin(temp_assign)].append(agent_permut[i]) #finally assign a_i in minimum valued t_j
                ########### Final Results
                greedy_sol = greedy_sol
                greedy_sol_val = solution_value(greedy_sol)
                greedy_sol_count = task_satisfy_count(greedy_sol)
                
                return greedy_sol, greedy_sol_val, greedy_sol_count

            ## Function Run for Final Results 
            ### Greedy for SCSGA-MF 
            greedy_start = time.time()
            greedy_sol, greedy_val, greedy_count = greedy()
            greedy_end = time.time()
            greedy_time = (greedy_end - greedy_start)
            
            
            
            ################# ALL FINAL RESULTS PRINT ###################
            rows_name = [datanames[t]]
            result_data = {
                'Value' : [greedy_val],
                'Count' : [greedy_count],
                'Time' : [greedy_time]
            }
            result.append(pd.DataFrame(result_data, index = rows_name))

                        
            
    print(datanames[t],"Completed")
    
    result=pd.concat(result)
    result.to_csv(result_dir_path+datanames[t]+'_greedy.csv')
    result.to_csv(result_dir_path+'greedy.csv', mode='a')
    result=[]
