'''
for continuous drag
2024-02-26
'''

import torch
import copy
from tqdm import tqdm
import numpy as np
pdist = torch.nn.PairwiseDistance(p=2)



def judge_edge(point_tuple,invert_code_d):
    y,x = point_tuple[0],point_tuple[1]
    max_y,max_x = invert_code_d.shape[2],invert_code_d.shape[3]
    y = 0 if y<0 else y
    x = 0 if x<0 else x
    y = int(max_y-1) if y>max_y-1 else y
    x = int(max_x-1) if x>max_x-1 else x
    new_point_tuple = (y,x)
    return new_point_tuple


def draw_arrow_graph_with_bg(points,vectors,img_path):
    import matplotlib.pyplot as plt
    import numpy as np
    
    background = plt.imread(img_path)
    fig, ax = plt.subplots()    
    ax.scatter(points[:, 0], points[:, 1], color='red') 
    for point, vector in zip(points, vectors):
        ax.arrow(point[0], point[1], vector[0], vector[1], head_width=1, head_length=2, fc='blue', ec='black')
    ax.axis('off')
    plt.grid(False)  
    plt.savefig('./arrow_chart_v2.png') 

def get_rectangle(mask: torch.Tensor):
    # get the rect of 1 in mask
    N,X,H,W = mask.shape  # eg:torch.Size([1, 1, 64, 64])   mask_cp_handle
    index_1 = torch.nonzero(mask)   
    min_y,min_x = torch.min(index_1,dim=0)[0][-2:]
    max_y,max_x = torch.max(index_1,dim=0)[0][-2:]
    left_top = torch.Tensor((min_y, min_x)).to(device=mask.device)
    left_bottom = torch.Tensor((min_y, max_x)).to(device=mask.device)
    right_top = torch.Tensor((max_y, min_x)).to(device=mask.device)
    right_bottom = torch.Tensor((max_y, max_x)).to(device=mask.device)
    rect = torch.stack((left_top, left_bottom, right_top, right_bottom),dim=0).to(device=mask.device)
    return rect, left_top, left_bottom, right_top, right_bottom
    
def interpolation(x):
    assert x.dim() == 4, "Input tensor x should have shape (1, C, N, M)"
    batch_size, channels, N, M = x.shape 

    for b in range(batch_size):
        zero_positions = (x[b, 0] == 0)

        for i in range(N):
            for j in range(M):
                if zero_positions[i, j]:
                    values = []  
                    weights = [] 

                    for k in range(1, j + 1):
                        if j - k >= 0 and x[b, 0, i, j - k] != 0:
                            values.append(x[b, :, i, j - k])
                            weights.append(1 / k)
                            break

                    for k in range(1, M - j):
                        if j + k < M and x[b, 0, i, j + k] != 0:
                            values.append(x[b, :, i, j + k])
                            weights.append(1 / k)
                            break

                    for k in range(1, i + 1):
                        if i - k >= 0 and x[b, 0, i - k, j] != 0:
                            values.append(x[b, :, i - k, j])
                            weights.append(1 / k)
                            break

                    for k in range(1, N - i):
                        if i + k < N and x[b, 0, i + k, j] != 0:
                            values.append(x[b, :, i + k, j])
                            weights.append(1 / k)
                            break

                    if weights:
                        total_weight = sum(weights)
                        interpolated_value = sum(w * v for w, v in zip(weights, values)) / total_weight
                        x[b, :, i, j] = interpolated_value

    return x

def get_circle(mask: torch.Tensor):
    rect, left_top, left_bottom, right_top, right_bottom = get_rectangle(mask=mask)
    center = torch.Tensor(((left_top[0] + right_bottom[0]) / 2, (left_top[1] + right_bottom[1]) / 2)).to(device=mask.device)  # y,x
    radius = pdist(center, left_top) 
    return center,radius


def get_scale_factor(C, A, OA, d_OA, R, O):
    '''
    xA, yA = A  xB, yB = B  xC, yC = C
    尝试是否可以同时作用于所有点
    '''
    # print("\n=============================")
    AC =  C-A  
    d_AC = torch.norm(AC)
    e_AC = AC/d_AC
    # print(f"O:{O}   \nA:{A} \nC:{C}  \nAC:{AC} \nd_AC:{d_AC} \ne_AC:{e_AC}  \nOA:{OA} \ntorch.dot(AC, OA):{torch.dot(AC, OA)}")
    L0 = torch.dot(AC, OA) / d_AC             #  |GA|    θ>90，L0<0
    L1 = torch.sqrt(R**2 - d_OA**2 + L0**2)   #  |GP|
    AP = (L1-L0)*e_AC  # GP-GA GP = L*t_AC GA = L0*t_AC
    PC = AC-AP
    # print(f"L0:{L0} \nL1:{L1}  \nAP:{AP} \nPC:{PC}")
    scale_factor = torch.norm(PC)/torch.norm(AP)  # |PC|/|AP| == |CD|/|AB|
    return scale_factor

def transform_point(point, shift_yx, scale_factor):
    shift_yx = shift_yx*scale_factor
    point_new = torch.round(point+shift_yx)
    return point_new


# 多对点drag，插值
def drag_stretch_multipoint_ratio_interp(invert_code,handle_points,target_points,mask_cp_handle,shift_yx=None,fill_mode='interpolation'):
    invert_code_d = copy.deepcopy(invert_code)
    if fill_mode == 'ori':
        pass
    if fill_mode == '0':
        invert_code_d[(mask_cp_handle>0).repeat(1,4,1,1)] = 0
    if fill_mode == "interpolation":
        invert_code_d[(mask_cp_handle>0).repeat(1,4,1,1)] = 0 
    if fill_mode == "random":
        invert_code_d[(mask_cp_handle>0).repeat(1,4,1,1)] = torch.rand_like(invert_code_d)[(mask_cp_handle>0).repeat(1,4,1,1)].to(device=invert_code_d.device)

    index_1 = torch.nonzero(mask_cp_handle) 
    O,R = get_circle(mask_cp_handle)   
    move_vectors = []       
    move_vectors_radio = [] 
    for point_i in range(len(handle_points)):
        A = handle_points[point_i].to(device=mask_cp_handle.device)/4    # y,x
        B = target_points[point_i].to(device=mask_cp_handle.device)/4
        shift_yx = B-A
        OA =  A-O  
        d_OA = torch.norm(OA)  # &
        for j, index in enumerate(tqdm(index_1, desc="get factor")):
            C = index[-2:]      # y,x
            scale_factor = get_scale_factor(C, A, OA, d_OA, R, O)
            move_vector = scale_factor*shift_yx
            if len(move_vectors)<=j:
                move_vectors.append([move_vector,])
                move_vectors_radio.append([1/(torch.norm(C-A)+0.0001),])
            else:
                move_vectors[j].append(move_vector)
                move_vectors_radio[j].append(1/(torch.norm(C-A)+0.0001))
    
    for j, index in enumerate(index_1):
        move_vectors[j] = torch.cat([ts.unsqueeze(0) for ts in move_vectors[j]], dim=0)
        move_vectors_radio[j] = torch.cat([ts.unsqueeze(0) for ts in move_vectors_radio[j]], dim=0)
    move_mode = "not recover"
    point_new_l = []
    point_new_l_value = {}
    graph_points = []
    graph_vectors = []
    heatmap_value = np.zeros((64,64))+20
    heatmap_value_target = np.zeros((64,64))+20
    flag = 0
    for j, index in enumerate(tqdm(index_1, desc="drag stretch")):
        C = index[-2:]
        radio_factor = move_vectors_radio[j]/move_vectors_radio[j].sum()
        move_vector = (radio_factor*move_vectors[j].T).T.sum(dim=0)
        point_new = torch.round(C+move_vector)

        # for draw arrow chart
        if flag%20 == 0:
            try:
                graph_points.append([int(torch.round(C[1])),int(torch.round(C[0]))])
                graph_vectors.append([int(torch.round(move_vector[1])),int(torch.round(move_vector[0]))])
                flag+=1
            except Exception as e:
                print(f"has a err: {e}")
                print(f"move_vector: {move_vector} C: {C} radio_factor: {radio_factor}  move_vectors_radio[j]: {move_vectors_radio[j]}")
        else:
            flag+=1
        
        try:
            point_tuple = (int(torch.round(point_new[0])),int(torch.round(point_new[1])))
            point_tuple = judge_edge(point_tuple,invert_code_d)
        except Exception as e:
            print(f"has a err: {e}")
            print(f"point_new: {point_new} C: {C} move_vector: {move_vector}")
        
            

        if move_mode == "not recover": # not recover point which has been cover
            if point_tuple in point_new_l:
                continue    
            point_new_l.append(point_tuple)
            invert_code_d[:,:,point_tuple[0],point_tuple[1]] = invert_code[:,:,int(torch.round(C[0])),int(torch.round(C[1]))]
        elif move_mode == "mean when recover": # when point is recovered, set the mean value to this point
            move_value = invert_code[:,:,int(torch.round(C[0])),int(torch.round(C[1]))]
            if point_tuple not in point_new_l_value.keys(): 
                point_new_l_value[point_tuple] = [move_value]
            else:
                point_new_l_value[point_tuple].append(move_value)

            if point_tuple in point_new_l:
                invert_code_d[:,:,point_tuple[0],point_tuple[1]] = sum(point_new_l_value[point_tuple])/len(point_new_l_value[point_tuple])
                continue
            point_new_l.append(point_tuple)
            invert_code_d[:,:,point_tuple[0],point_tuple[1]] = invert_code[:,:,int(torch.round(C[0])),int(torch.round(C[1]))]  

    if fill_mode == "interpolation":
        invert_code_d = interpolation(invert_code_d)
    return invert_code_d  





