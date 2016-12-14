import os
from collections import defaultdict
import csv
import numpy as np

model_list = os.listdir('../Super_models')
print(len(model_list))
results = defaultdict(list) 
count = 0
for model_dir in model_list:
#    if model_dir == 'POS_tagging_cap1_num1_bi1_numlayers1_embeddim300_embedtypeglovevector_seqlength-1_seed0_longskip1_units64_lrate0.01_normalize0_dropout1.0_inputdp1.0_embeddingtrain0_suffix1_windowsize1':
#	print('working')
#    else:
#	continue 	
    model_configs = model_dir.split('_')
    if len(model_configs) <= 4:
        continue
    numlayers = model_configs[5] 
    embeddim =  model_configs[6]
    embedtype = model_configs[7] 
    units = model_configs[11]
    embeddingtrain = model_configs[16]
    windowsize = model_configs[18]
    model_results = os.listdir(os.path.join('..', 'Super_models', model_dir))
    max_epoch = 0
    accuracy = 0

    for model_result in model_results:
        if 'meta' in model_result:
            result = model_result.split('_')
            if len(result) == 3: 
                epoch = float(result[0][5:])
                if max_epoch < epoch:
                    max_epoch = epoch
                    accuracy = float(result[1][8:])
                    unknown_accuracy = float(result[2][7:14])
		    
    if accuracy>0:
	if accuracy > 0.90:
	    print(model_dir, accuracy, max_epoch)

	count+= 1
		
		

        results[(numlayers, embeddim, embedtype, units, embeddingtrain, windowsize)].append((accuracy, unknown_accuracy))

#print(count)

result_table = np.zeros((2, 4, 3, 2, 2, 4))

#print(results)
with open('../results_table.csv', 'wb') as csvfile:
    mat_writer=csv.writer(csvfile, delimiter=',')
    for config, accuracies in results.items():
	average_accuracy = np.mean([accuracy[0] for accuracy in accuracies])
	#mat_writer.writerow(['config'] + list(config))	
	#mat_writer.writerow(['average', average_accuracy])
	#
	#unknown_accuracy = np.mean([accuracy[1] for accuracy in accuracies])

	#mat_writer.writerow(['unknown', unknown_accuracy])
	for accuracy in accuracies:
		known = accuracy[0]
		unknown = accuracy[1]
		mat_writer.writerow([known, unknown]+ list(config))
	
	
#print(len(results))
    


   
    
    


