import numpy as np
import os


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
distribution_inst=con[3]
confile.close()

variance=[0.25, 0.4, 0.5, 0.75, 1]

for p in range (distribution_inst):
    for v in range(len(variance)):
        data_weight=variance[v]*(np.ceil(n_agents/m_tasks))
        
        dir_path = './dataset/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        ##### UPD #######
        upd_agents=np.random.uniform(0, 1, (n_agents,features))
        upd_tasks=np.random.uniform(0, data_weight, (m_tasks,features))
        upd = np.concatenate((upd_agents,upd_tasks),axis=0)
        #np.save(os.path.join('dataset', 'upd'+str(p)+'_v'+str(v)+'.npy'), upd)
        np.save(dir_path+'upd'+str(p)+'_v'+str(v)+'.npy', upd)
        