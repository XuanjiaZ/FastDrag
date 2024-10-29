import torch
import copy

def save_tensor(path,x):
    print(f'save {path}')
    with open(path, "w") as file:
        for i in range(x.shape[2]):
            s = ''
            for j in range(x.shape[3]):
                s += str(int(x[0][0][i][j]))+' '
            file.write(s)
            file.write('\n')


def get_mask_of_point(template, point, R=5, flag = 0):
    # get mask_cp_target 
    # template: torch.Size([1, 1, 64, 64])
    # point: tensor([   7., -104.])
    # 1 is masked, ie. need move
    mask = torch.zeros_like(template)
    x,y = point[1], point[0]
    mask[:,:,int(y-R):int(y+R),int(x-R):int(x+R)] = 1
    return mask

def get_complementary_of_mask(mask_target, mask_handle, flag = 0):
    # get mask_cp_handle ie need paint      complementary
    mask_handle = mask_handle-mask_target
    mask_handle = torch.where(mask_handle>=0,mask_handle,0.)     #  ie: if >=0  mask = mask_handle  else 0
    return mask_handle


def shift_matrix(matrix, x, y):
    # matrix shape should be (1,n,x,y)

    mtype = matrix.dtype
    if x > 0:
        matrix = torch.cat([torch.zeros(matrix.size(0),matrix.size(1),matrix.size(2), x).to(device=matrix.device,dtype=matrix.dtype), matrix[:,:,:, :-x]], dim=3)
    elif x < 0:
        matrix = torch.cat([matrix[:,:,:, -x:], torch.zeros(matrix.size(0),matrix.size(1),matrix.size(2), -x).to(device=matrix.device,dtype=matrix.dtype)], dim=3)

    if y > 0:
        matrix = torch.cat([torch.zeros(matrix.size(0),matrix.size(1),y, matrix.size(3)).to(device=matrix.device,dtype=matrix.dtype), matrix[:,:,:-y]], dim=2)
    elif y < 0:
        matrix = torch.cat([matrix[:,:,-y:], torch.zeros(matrix.size(0),matrix.size(1),-y, matrix.size(3)).to(device=matrix.device,dtype=matrix.dtype)], dim=2)

    return matrix.to(dtype=mtype)

def copy_past(ori_x,mask_target,shift_yx):
    '''
    ori_x shape: (n,m,x,y)  x=y=64
    mask_target shape: (1,1,x,y)
    shift_yx: (delta y, delta x)
    '''
    shift_y,shift_x = int(shift_yx[0]/4),int(shift_yx[1]/4)
    dype_t = ori_x.dtype
    ori_x_d = copy.deepcopy(ori_x)
    x_cp_target = shift_matrix(ori_x_d,shift_x,shift_y)*mask_target
    ori_x = ori_x*(1-mask_target) + x_cp_target
    ori_x = ori_x.to(dtype=dype_t)
    return ori_x


def part_to_all(ori_x,target_point):
    '''
    ori_x shape: (n,m,x,y)  x=y=64
    target_point: y,x
    v0: mask_part: A square with a side length 2*R centered on the target point
    '''
    R = 6
    print("R=",R)
    maxp_h = int(ori_x.shape[2]) # 42
    maxp_w = int(ori_x.shape[3]) # 64

    y0 = (target_point-R).to(dtype=torch.int)[1] if ((target_point-R)[1]>=0).all() else 0
    y1 = (target_point+R).to(dtype=torch.int)[1] if ((target_point+R)[1]<maxp_w).all() else maxp_w
    x0 = (target_point-R).to(dtype=torch.int)[0] if ((target_point-R)[0]>=0).all() else 0
    x1 = (target_point+R).to(dtype=torch.int)[0] if ((target_point+R)[0]<maxp_h).all() else maxp_h

    rep_h = int((maxp_h/(x1-x0))+1)
    rep_w = int((maxp_w/(y1-y0))+1)
    all_x = ori_x[:,:,x0:x1,y0:y1].repeat(1,1,rep_h,rep_w)
    all_x = all_x[:,:,:maxp_h,:maxp_w]

    return all_x


def paint_past(invert_code,invert_code_d,mask_cp_handle,mask_cp_target,target_point=None, mask_fill=None):
    if target_point==None:
        all_x = invert_code_d
    elif mask_fill!=None:
        index_1 = torch.nonzero(mask_fill)   
        print(len(index_1))
        if len(index_1) == 0:
            all_x = part_to_all(invert_code_d,target_point)
        else:
            min_y,min_x = torch.min(index_1,dim=0)[0][-2:]
            max_y,max_x = torch.max(index_1,dim=0)[0][-2:]
            all_x = part_to_all(invert_code_d,torch.Tensor([(min_y+max_y)/2, (min_x+max_x)/2]))
            print("fill")
    else:
        all_x = part_to_all(invert_code_d,target_point)
    invert_code = invert_code*(1-mask_cp_handle) + all_x*mask_cp_handle
    invert_code = invert_code.to(dtype=invert_code_d.dtype)
    return invert_code


if __name__ == '__main__':
    ori_x = torch.arange(64).view(8,8)
    ori_x = torch.unsqueeze(torch.unsqueeze(ori_x,dim=0),dim=0)
    target_point = torch.Tensor([4,4])
    print(ori_x.shape,'\n',ori_x)
    all_x = part_to_all(ori_x,target_point)
    print(all_x.shape,'\n',all_x)