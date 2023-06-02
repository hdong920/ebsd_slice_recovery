import torch

def dist(x, y):
    return torch.linalg.norm(x-y, ord=2, dim=-1)

def project(est, prev_ids, next_ids, id2cu, min_neighbors=6):
    '''
    Projection Algorithm to turn transformer outputs to IDs.
    Inputs: 
        est: cubochoric values (H, W, 3)
        prev_ids: previous slice's IDs (H, W)
        next_ids: next slice's IDs (H, W)
        id2cu: dictionary of IDs to cubochorics
        min_neighbors: minimum neighbors required to project a voxel. This number decrements if no voxels meet this requirement.
    Outputs:
        est_ids: predicted IDs 
    '''
    H, W, _ = est.shape
    est_ids = torch.zeros((H, W))
    assigned = torch.zeros((H, W)).bool()
    last_num_assigned = 0
    while last_num_assigned < assigned.numel():
        if last_num_assigned == torch.sum(assigned):
            min_neighbors -= 1
        last_num_assigned = torch.sum(assigned)
        for h in range(H):
            for w in range(W):
                if not assigned[h, w]:
                    if prev_ids[h, w] == next_ids[h, w]:
                        est_ids[h, w] = prev_ids[h, w]
                        assigned[h, w] = True
                    else:
                        neighbors = set()
                        num_neighbors = 0
                        neighbors.add(prev_ids[h, w].item())
                        neighbors.add(next_ids[h, w].item())
                        num_neighbors += 2
                        if h > 0 and assigned[h-1, w]:
                            neighbors.add(est_ids[h-1, w].item())
                            num_neighbors += 1
                        if h < H-1 and assigned[h+1, w]:
                            neighbors.add(est_ids[h+1, w].item())
                            num_neighbors += 1
                        if w > 0 and assigned[h, w-1]:
                            neighbors.add(est_ids[h, w-1].item())
                            num_neighbors += 1
                        if w < W-1 and assigned[h, w+1]:
                            neighbors.add(est_ids[h, w+1].item())
                            num_neighbors += 1
                            
                            
                        if num_neighbors >= min_neighbors:
                            best_d = float('inf')
                            for n in neighbors:
                                d = dist(id2cu[n], est[h, w])
                                if d < best_d:
                                    best_d = d
                                    est_ids[h, w] = n
                            assigned[h, w] = True
    return est_ids