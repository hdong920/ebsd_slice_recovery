import torch
import numpy as np
from torch import nn as nn 
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import time
import wandb 
import torchvision.transforms as T 

from model import Axial3DTransformer
from utils.rotlib import eu2cu, cu2eu 
from utils.write_slices_out_of_dream3d import DREAM3D
from utils.projection import project


torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

config = {
    'epochs': 40,
    'batches_per_epoch': 4000,
    'batch_size': 1,
    'dropout': 0.1,
    'ff_kernel': 3,
    'lx': 64,
    'ly': 7,
    'lz': 64,
    'hidden_dim': 128,
    'depth': 8,
    'heads': 8,
    'mask_num': 1, # untested when mask_num != 1
    'lr': 0.01,
    'weight_decay': 0.00001,
    'warmup_ratio': 0.05,
}

if config['mask_num'] != 1:
    raise NotImplementedError


voxel_arrays = ['EulerAngles', 'BoundaryCells', 'FeatureIds']
files = [
    "mu2_sig0-4_trans0",
    "mu2-5_sig0-4_trans0",
    "mu3_sig0-4_trans0",
    "mu2-3_sig0-4_trans0",
    "mu2-3_sig0-4_trans1",
    "mu2-3_sig0-4_trans2",
    "mu2-3_sig0-4_trans3",
    "mu2-3_sig0-4_trans4",
    "mu2-3_sig0-4_trans5",
]
train_paths = [f'../data/synthetic/train_{file_params}.dream3d' for file_params in files]
val_paths = [f'../data/synthetic/val_{file_params}_1.dream3d' for file_params in files]


def load_data(paths):
    compiled_data = []
    compiled_ids = []
    compiled_edges = []
    for path in paths:
        full_h5_data_set = DREAM3D(path, synthetic=True)
        data = full_h5_data_set.get_element_data_array(voxel_arrays[0])
        data = torch.Tensor(data).squeeze()
        data_shape = data.shape
        data = torch.Tensor(eu2cu(data))
        data = data.reshape(data_shape)
        ids = full_h5_data_set.get_element_data_array(voxel_arrays[2])
        ids = torch.Tensor(ids).squeeze()
        edges = full_h5_data_set.get_element_data_array(voxel_arrays[1])
        edges = torch.Tensor(edges).squeeze() > 0
        full_h5_data_set.close()

        compiled_data.append(data)
        compiled_ids.append(ids)
        compiled_edges.append(edges)

    compiled_data = torch.stack(compiled_data).permute((0, 4, 1, 2, 3))
    compiled_ids = torch.stack(compiled_ids)
    compiled_edges = torch.stack(compiled_edges)
    
    # mean and sd of all train volumes
    chan_means = torch.Tensor([ 0.0011, -0.0010,  0.0033]).reshape(1, 3, 1, 1, 1)
    chan_sds = torch.Tensor([0.5872, 0.5923, 0.5895]).reshape(1, 3, 1, 1, 1)
    compiled_data = (compiled_data - chan_means) / chan_sds
    
    return compiled_data, compiled_ids, compiled_edges

train_data, train_ids, train_edges = load_data(train_paths)
val_data, val_ids, val_edges = load_data(val_paths)
print("data loaded")


pad = 3
pad = (pad, pad, 0, 0, pad, pad)
train_data = F.pad(train_data, pad, mode='constant', value=0)
train_ids = F.pad(train_ids, pad, mode='constant', value=0)
train_edges = F.pad(train_edges, pad, mode='constant', value=0)
print("train cubochorics shape: ", train_data.shape)
print("train IDs shape: ", train_ids.shape)
print("train boundaries shape: ", train_edges.shape)

V = len(files)
lx = config['lx']
ly = config['ly']
lz = config['lz']
Nx = train_data.shape[2]
Ny = train_data.shape[3]
Nz = train_data.shape[4]

# Turn validation volumes into samples of shape (lx, ly, lz)
reshaped_val_data = []
reshaped_val_ids = []
reshaped_val_edges = []
for v in range(V):
    for x in range(0, lx*(val_data.shape[2]//lx), lx):
        for y in range(0, ly*(val_data.shape[3]//ly), ly):
            for z in range(0, lz*(val_data.shape[4]//lz), lz):
                reshaped_val_data.append(val_data[v, :, x:(x+lx), y:(y+ly), z:(z+lz)])
                reshaped_val_ids.append(val_ids[v, x:(x+lx), y:(y+ly), z:(z+lz)])
                reshaped_val_edges.append(val_edges[v, x:(x+lx), y:(y+ly), z:(z+lz)])

val_data = torch.stack(reshaped_val_data) # (243, 3, lx, ly, lz)
val_ids = torch.stack(reshaped_val_ids)
val_edges = torch.stack(reshaped_val_edges)

val_len_per_volume = len(val_data) // V
val_data = val_data.reshape((V, val_len_per_volume, 3, lx, ly, lz))
val_ids = val_ids.reshape((V, val_len_per_volume, lx, ly, lz))
val_edges = val_edges.reshape((V, val_len_per_volume, lx, ly, lz))

# list of masked indices
val_missing_mask_ind = np.random.choice(np.arange(1, ly-1), size=(V, val_len_per_volume, config['mask_num']), replace=True)
val_missing_mask_ind = torch.Tensor(val_missing_mask_ind).long()
val_missing_mask = 1 - nn.functional.one_hot(val_missing_mask_ind, ly).reshape((V, val_len_per_volume, 1, 1, ly, 1)) # 1 for observed, 0 else

# Create dictionaries of IDs to cubochorics
val_id2cu = [dict() for v in range(V)]
val_data = val_data.permute((0, 1, 3, 4, 5, 2))
for v in range(V):  
    max_id = int(val_ids[v].max().item())
    for i in range(1, max_id+1):
        try:
            val_id2cu[v][i] = val_data[v][val_ids[v] == i][0]
        except:
            # print(v, i)
            pass
val_data = val_data.permute((0, 1, 5, 2, 3, 4))

print("data preparation complete")

# unused but can be used for visualization
def idmap2cumap(m, id2cu):
    H, W = m.shape
    out = torch.zeros((H, W, 3))
    for i in range(H):
        for j in range(W):
            out[i, j] = id2cu[m[i, j].item()]
    return out


model = Axial3DTransformer(
    dim=config['hidden_dim'], 
    depth=config['depth'], 
    heads=config['heads'], 
    dropout=config['dropout'], 
    input_shape=(lx, ly, lz), 
    ff_kernel=config['ff_kernel']
).to(device)

model.eval()

criterion = nn.MSELoss(reduction='none')

optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=config['lr'], 
    steps_per_epoch=config['batches_per_epoch'], 
    epochs=config['epochs'], 
    pct_start=config['warmup_ratio']
)


batch_size = config['batch_size']

def get_batch(full_dataset, full_id_dataset, full_edge_dataset, transform=True):
    batch = torch.zeros((batch_size, 3, lx, ly, lz))
    batch_indices = np.random.choice(np.arange(0, full_dataset.shape[-2] - ly - 1), size=batch_size, replace=True)
    batch_missing_mask_ind = torch.Tensor(np.random.choice(np.arange(1, ly-1), size=(batch_size, config['mask_num']), replace=True)).to(dtype=torch.int64) # (B, mask_num)
    batch_missing_mask = 1 - nn.functional.one_hot(batch_missing_mask_ind, ly).reshape((batch.shape[0], 1, 1, batch.shape[3], 1)) # (B, 1, 1, ly, 1)
    batch_ids = torch.zeros((batch_size, lx, ly, lz))
    batch_edges = torch.zeros((batch_size, lx, ly, lz))
    for i in range(batch_size):
        volume = torch.randint(low=0, high=V, size=(1,))[0]
        dataset = full_dataset[volume]
        id_dataset = full_id_dataset[volume]
        edge_dataset = full_edge_dataset[volume]
        batch_element = dataset[:, :, batch_indices[i]:(batch_indices[i] + ly)] #(3, Nx, ly, Nz)
        batch_element_ids = id_dataset[:, batch_indices[i]:(batch_indices[i] + ly)] # (Nx, ly, Nz)
        batch_element_edges = edge_dataset[:, batch_indices[i]:(batch_indices[i] + ly)] # (Nx, ly, Nz)
        
        # Transformations
        if transform:
            random_x = np.random.choice(np.arange(0, Nx - lx))
            random_z = np.random.choice(np.arange(0, Nz - lz))
            batch_element = TF.crop(batch_element.permute((2, 0, 1, 3)), random_x, random_z, height=lx, width=lz) #(ly, 3, lx, lz)
            batch_element_ids = TF.crop(batch_element_ids.permute((1, 0, 2)), random_x, random_z, height=lx, width=lz) #(ly, lx, lz)
            batch_element_edges = TF.crop(batch_element_edges.permute((1, 0, 2)), random_x, random_z, height=lx, width=lz) #(ly, lx, lz)
            angle = int(np.random.choice([0, 90, 180, 270]))
            batch_element = TF.rotate(batch_element, angle=angle) 
            batch_element_ids = TF.rotate(batch_element_ids, angle=angle, fill=-float('inf')) 
            batch_element_edges = TF.rotate(batch_element_edges, angle=angle, fill=-float('inf')) 
            if torch.rand(1) < 0.5:
                batch_element = TF.hflip(batch_element)
                batch_element_ids = TF.hflip(batch_element_ids)
                batch_element_edges = TF.hflip(batch_element_edges)
            if torch.rand(1) < 0.5:
                batch_element = -batch_element
            
            # Random linear transformation
            batch_element = batch_element * (torch.rand(3).reshape(1, 3, 1, 1) + 0.5) + (torch.rand(3).reshape(1, 3, 1, 1) - 0.5)
            batch_element = batch_element.permute((1, 2, 0, 3))
            batch_element_ids = batch_element_ids.permute((1, 0, 2))
            batch_element_edges = batch_element_edges.permute((1, 0, 2))
            
        batch[i] = batch_element
        batch_ids[i] = batch_element_ids
        batch_edges[i] = batch_element_edges
        
    return batch, batch_ids, batch_edges, batch_missing_mask_ind, batch_missing_mask.to(dtype=torch.int8)


def train(model, epoch):
    model.train()
    total = 0
    train_loss = 0.0
        
    for batch_i in range(config['batches_per_epoch']):
        # Randomly swap dims of data
        perm = 2 + torch.randperm(3)
        rotated_data = train_data.permute((0, 1, perm[0], perm[1], perm[2]))
        perm = perm-1
        rotated_ids = train_ids.permute((0, perm[0], perm[1], perm[2]))
        rotated_edges = train_edges.permute((0, perm[0], perm[1], perm[2]))
        
        optimizer.zero_grad()
        
        
        batch, _, batch_edges, batch_missing_mask_ind, batch_missing_mask = get_batch(rotated_data, rotated_ids, rotated_edges, transform=True)
        batch = batch.to(device)
        batch_edges = batch_edges.to(device)
        batch_missing_mask = batch_missing_mask.float().to(device)
        
        inv_mask = 1 - batch_missing_mask
        batch_input_cu = batch * batch_missing_mask
        loss_mask = (batch_edges * inv_mask.squeeze(1)).long().unsqueeze(1)
        
        output = model(batch_input_cu)
        output_pred = output[:, :, :, batch_missing_mask_ind[:, 0]]
        target_pred = batch[:, :, :, batch_missing_mask_ind[:, 0]]
        loss_mask = batch_edges[:, :, batch_missing_mask_ind[:, 0]]
        loss = torch.sum(criterion(output_pred, target_pred) * loss_mask.unsqueeze(1) / (torch.sum(loss_mask) + 1e-6))
            
        
        train_loss += loss.item()
        loss.backward() 
        optimizer.step() 
        scheduler.step()
 
        if batch_i % 200 == 199:
            print(f"Train loss after {batch_i+1} batches: ", train_loss/(batch_i + 1))
       
    return train_loss/config['batches_per_epoch']


def eval(model):
    model.eval()
    
    correct = 0
    total = 0
    edge_correct = 0
    edge_total = 0
    loss = 0.0
    count = 0
    for v in range(V):
        for i in range(0, val_len_per_volume, batch_size):
            with torch.inference_mode():

                batch = val_data[v, i:min(i+batch_size, val_len_per_volume)]
                batch_ids = val_ids[v, i:min(i+batch_size, val_len_per_volume)]
                batch_edges = val_edges[v, i:min(i+batch_size, val_len_per_volume)]
                batch_missing_mask_ind = val_missing_mask_ind[v, i:min(i+batch_size, val_len_per_volume)]
                batch_missing_mask = val_missing_mask[v, i:min(i+batch_size, val_len_per_volume)]


                batch = batch.to(device)
                batch_edges = batch_edges.to(device)
                batch_missing_mask = batch_missing_mask.float().to(device)


                inv_mask = 1 - batch_missing_mask
                batch_input_cu = batch * batch_missing_mask
                loss_mask = (batch_edges * inv_mask.squeeze(1)).long().unsqueeze(1)

                output = model(batch_input_cu)
                output_pred = output * loss_mask
                target_pred = batch * loss_mask

                loss += torch.sum(criterion(output_pred, target_pred) / (torch.sum(loss_mask) + 1e-6)).item()


            ### Projections
            for j in range(len(output)):
                count += 1
                missing_ind = batch_missing_mask_ind[j][0].item()
                input_img_cu = batch_input_cu[j, :, :, missing_ind].permute((1, 2, 0))
                loss_mask_j = loss_mask[j,:, :, missing_ind].bool().squeeze(0).cpu()
                target_img = batch[j, :, :, missing_ind].permute((1, 2, 0))
                output_img = output[j, :, :, missing_ind].permute((1, 2, 0))
                
                prev_ids = batch_ids[j, :, missing_ind-1]
                target_ids = batch_ids[j, :, missing_ind]
                next_ids = batch_ids[j, :, missing_ind+1]
                
                projected = project(output_img.cpu(), prev_ids, next_ids, val_id2cu[v], min_neighbors=6)

                correct += (torch.sum(projected == target_ids)/ projected.numel()).item()
                edge_correct += torch.sum((projected == target_ids) * loss_mask_j).item() / torch.sum(loss_mask_j).item()
                

            del batch, batch_ids, batch_missing_mask
            torch.cuda.empty_cache()

    ### Calculate Accuracy
    accuracy = correct / (V * val_len_per_volume)
    edge_accuracy = edge_correct / (V * val_len_per_volume)
    loss /= (V * val_len_per_volume)
    return loss, edge_accuracy*100, accuracy *100




print("INITIALIZATION")
val_loss, val_edge_acc, val_acc = eval(model)
print("val loss: ", val_loss)
print("val acc: ", val_acc)
print("val edge acc: ", val_edge_acc)

print("Starting Training")

best_val_acc = 0
best_model = None
    


for epoch in range(1, config['epochs'] + 1):
    print("EPOCH ", epoch)
    train_loss = train(model, epoch)
    val_loss, val_edge_acc, val_acc = eval(model)
    print("val loss: ", val_loss)
    print("val acc: ", val_acc)
    print("val edge acc: ", val_edge_acc)
    wandb.log({
        "train loss": train_loss, 
        "validation loss": val_loss, 
        "validation edge accuracy": val_edge_acc, 
        "validation accuracy": val_acc, 
        'lr': scheduler.get_last_lr()[0]
    })


    if best_val_acc < val_edge_acc:
        torch.save(model.state_dict(), f'./checkpoints/transformer.pth')
        best_val_acc = val_edge_acc
