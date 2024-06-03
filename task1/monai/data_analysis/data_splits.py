#%%
import json

# Path to your JSON file
filename = '/data/blobfuse/default/autopet2024/data/splits_final.json'
with open(filename, 'r') as file:
    data = json.load(file)

# %%
data_new = [{}, {}, {}, {}, {}]
for i in range(5):
    data_i = data[i]
    data_i_train = data_i['train']
    data_i_valid = data_i['val']

    data_i_train_new = []
    data_i_valid_new = []

    for idx, id in enumerate(data_i_train):
        if id.startswith('fdg'):
            id_info = id.split('_')
            new_id = f'{id_info[0]}_{id_info[1]}_{id_info[2][:10]}'
            data_i_train_new.append(new_id)
        else:
            data_i_train_new.append(id)

    for idx, id in enumerate(data_i_valid):
        if id.startswith('fdg'):
            id_info = id.split('_')
            new_id = f'{id_info[0]}_{id_info[1]}_{id_info[2][:10]}'
            data_i_valid_new.append(new_id)
        else:
            data_i_valid_new.append(id)

    data_new[i]['train'] = data_i_train_new
    data_new[i]['val'] = data_i_valid_new

# %%
filename = 'data_split.json'
with open(filename, 'w') as file:
    json.dump(data_new, file, indent=4)
# %%
