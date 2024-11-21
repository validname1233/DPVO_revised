import torch
import torch.nn as nn
import torch.nn.init as init
import cv2
import numpy as np
autocast = torch.cuda.amp.autocast

class LF_Net(nn.Module): #
    def __init__(self,input):
        super(LF_Net, self).__init__()

        dtype = input.dtype
        batch = input.shape[0]
        height = input.shape[3]
        width = input.shape[4]
        channel = input.shape[2]

        #building_block
        self.batch_norm1 = nn.BatchNorm2d(num_features=16, momentum=0.9, affine=True, eps=1e-5)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv_layer1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2)
        init.xavier_uniform_(self.conv_layer1.weight)
        self.batch_norm2 = nn.BatchNorm2d(num_features=16, momentum=0.9, affine=True, eps=1e-5)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv_layer2=nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2)
        init.xavier_uniform_(self.conv_layer2.weight)

        #get_model
        self.conv_layer_get_model = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        #xavier初始化
        init.xavier_uniform_(self.conv_layer_get_model.weight)
        self.batch_norm_get_model = nn.BatchNorm2d(num_features=16, momentum=0.9, affine=True, eps=1e-5)
        
        self.conv_layer3=nn.Conv2d(in_channels=16, out_channels=1, kernel_size=5, stride=1, padding=2)
        init.xavier_uniform_(self.conv_layer3.weight)

        #soft_nms_3d
        ksize_nms=15
        num_scales=channel
        self.pad_h_left, _ = calculate_padding(ksize_nms, 1, height)
        self.pad_w_left, _ = calculate_padding(ksize_nms, 1, width)
        self.conv3d = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(num_scales, ksize_nms, ksize_nms), stride=(num_scales,1,1), padding=(0,self.pad_h_left,self.pad_w_left))
        sum_filter = torch.ones((1, 1, num_scales, ksize_nms, ksize_nms))
        with torch.no_grad():  #咱还不知道这个权重要不要在反向传播中改变
            self.conv3d.weight.copy_(sum_filter)

    @autocast(True)
    def forward(self, input):
        #curr_in=input.squeeze(1) #转换为（B，C，H，W）
        #print("input",input.shape)
        kpts=self.build_from_features(input)
        #kpts=kpts.unsqueeze(0)
        #print(kpts.shape)
        return kpts
    
    def building_block(self,input):    #建立残差块
        curr_in=input
        #因为是残差块所以建立切片
        shortcut=curr_in

        #_BATCH_NORM_DECAY =  0.9  _BATCH_NORM_EPSILON = 1e-5  axis = 1
        #批量归一化+激活函数
        
        curr_in = self.batch_norm1(curr_in)
        curr_in=self.leaky_relu1(curr_in)
        
        #卷积+批量归一化
        curr_in =self.conv_layer1(curr_in)

        curr_in = self.batch_norm2(curr_in)
        curr_in=self.leaky_relu2(curr_in)

        #再次卷积
        
        curr_in =self.conv_layer2(curr_in)

        return curr_in+shortcut

    def get_model(self,input):
        num_conv = 0
            
        curr_in=input

        #print(input.shape)
        
        #第一次卷积，全卷积网络 因为卷积核是5x5，步幅为1，故填充1格
        curr_in = self.conv_layer_get_model(curr_in)
        num_conv += 1

        #残差块
        for i in range(3):
            curr_in = self.building_block(curr_in)
            num_conv += 2

        curr_in = self.batch_norm_get_model(curr_in)
        
        #特征图
        feat_maps = curr_in.clone

        

        max_scale=1.4142135623730951
        min_scale=0.7071067811865475
        num_scales=5
        scale_log_factors = np.linspace(np.log(max_scale), np.log(min_scale), num_scales)
        scale_factors = np.exp(scale_log_factors)

        score_maps_list = []

        base_height_f = curr_in.shape[2]
        base_width_f = curr_in.shape[3]

        for i, s in enumerate(scale_factors):
                inv_s = 1.0 / s # scale are defined by extracted patch size (s of s*default_patch_size) so we need use inv-scale for resizing images
                feat_height = int((base_height_f * inv_s+0.5))
                feat_width = int((base_width_f * inv_s+0.5))
                rs_feat_maps = nn.functional.interpolate(curr_in, size=(feat_height, feat_width), mode='bilinear', align_corners=False)      
                #卷积生成得分图
                
                score_maps = self.conv_layer3(rs_feat_maps)
                score_maps_list.append(score_maps)
        num_conv += 1

        endpoints = {}
        endpoints['scale_factors'] = scale_factors
        endpoints['feat_maps'] = feat_maps
        endpoints['pad_size'] = num_conv * (5//2) #5是conv_ksize

        

        return score_maps_list, endpoints

    def instance_normalization(self,inputs):
        # normalize 0-mean, 1-variance in each sample (not take batch-axis)
        inputs_dim = inputs.dim()  # 获取输入张量的维度数
        
        var_eps = 1e-3  # 用于归一化计算中的 epsilon，防止除零错误
        
        if inputs_dim == 4:
            moments_dims = (2, 3)  # NCHW格式，计算高度和宽度维度上的均值和方差
        elif inputs_dim == 2:
            moments_dims = (1,)  # 对二维张量，仅计算第二个维度上的均值和方差
        else:
            raise ValueError(f'instance_normalization suppose input dim is 4 or 2: inputs_dim={inputs_dim}\n')
        
        inputs=inputs.float()

        # 计算均值和方差
        mean = inputs.mean(dim=moments_dims, keepdim=True)
        variance = inputs.var(dim=moments_dims, keepdim=True, unbiased=False)
        
        # 实现非参数化归一化
        outputs = (inputs - mean) / torch.sqrt(variance + var_eps)
        
        return outputs

    def soft_nms_3d(self,scale_logits, ksize, com_strength):
        # apply softmax on scalespace logits
        # scale_logits: [B,S,H,W]
        #print("scale_logits",scale_logits.shape)
        num_scales = scale_logits.shape[1] #得到S

        scale_logits_d = scale_logits.unsqueeze(1)  # [B,1,S,H,W] in order to apply pool3d 1其实是灰度图的通道数
        #print("scale_logits_d",scale_logits_d.shape)
        #padding = (pad_w_left, pad_w_right, pad_h_left, pad_h_right, pad_d_left, pad_d_right)
        #max_maps = nn.functional.max_pool3d(scale_logits_d,(num_scales,ksize,ksize),(num_scales,1,1),padding=(pad_w_left,pad_h_left,0))
        scale=scale_logits_d.shape[2]
        max_maps=nn.functional.max_pool3d(scale_logits_d,(scale,ksize,ksize),(scale,1,1),padding=(0,self.pad_h_left,self.pad_w_left))
        
        max_maps = max_maps.squeeze(1)
        exp_maps = torch.exp(com_strength * (scale_logits-max_maps))
        exp_maps_d = exp_maps.unsqueeze(1) # [B,1,S,H,W]
        
        #print(exp_maps_d.shape,exp_maps_d.device)
        #conv3d.bias.data=conv3d.bias.data.half()
        
        #print(torch.cuda.current_device())

        sum_ex=self.conv3d(exp_maps_d)
        #print("exp_maps:",exp_maps.shape)
        sum_ex = sum_ex.squeeze(1)  # [B,S,H,W]
        #print("sum_ex:",sum_ex.shape)
        probs = exp_maps / (sum_ex + 1e-6)    

        return probs

    def soft_max_and_argmax_1d(self,inputs_t,inputs_index=None,dim=1,com_strength1=100.0,com_strength2=100.0):
        #safe softmax

        max_value1,_=torch.max(inputs_t, dim=dim, keepdim=True)
        inputs_exp1 = torch.exp(com_strength1*(inputs_t - max_value1))
        inputs_softmax1 = inputs_exp1 / (torch.sum(inputs_exp1, dim=dim, keepdim=True) + 1e-8)

        inputs_exp2 = torch.exp(com_strength2*(inputs_t - max_value1))
        inputs_softmax2 = inputs_exp2 / (torch.sum(inputs_exp2, dim=dim, keepdim=True) + 1e-8)

        inputs_max = torch.sum(inputs_t * inputs_softmax1, dim=dim, keepdim=False)
        #Hadamard product，又称为元素逐个相乘（element-wise product）或逐元素积
        

        inputs_index_shp = [1,]*len(inputs_t.shape)
        inputs_index_shp[dim] = -1
        inputs_amax=None

        # 重新调整 inputs_index 的形状
        if(inputs_index!=None):
            inputs_index = inputs_index.reshape(inputs_index_shp)  # 或者使用 inputs_index.view(inputs_index_shp)
        # 计算加权和，沿着指定的 axis 维度
        
            inputs_amax = torch.sum(inputs_index * inputs_softmax2, dim=dim, keepdim=False)

        return inputs_max, inputs_amax

    def end_of_frame_masks(self,height, width, radius, dtype=torch.float32):  
        #函数 end_of_frame_masks 创建一个掩码，表示一个矩形区域，其中中间部分是 1（有效区域），而周围的部分通过填充的方式被扩展成 0（无效区域）。
        
        # 创建一个形状为 [1, height - 2*radius, width - 2*radius, 1] 的全1张量
        eof_masks = torch.ones((1, 1,height - 2 * radius, width - 2 * radius), dtype=dtype)
        
        # 在指定的维度上填充0 表示边缘区域不管
        eof_masks = nn.functional.pad(eof_masks, pad=(radius, radius, radius, radius))   
        
        return eof_masks

    def non_max_suppression(self,inputs, thresh=0.0, ksize=5, dtype=torch.float32, name='NMS'):
        dtype = inputs.dtype
        batch = inputs.shape[0]
        height = inputs.shape[2]
        width = inputs.shape[3]
        channel = inputs.shape[1]
        hk = ksize // 2
        zeros = torch.zeros_like(inputs)
        works = torch.where(inputs < thresh, zeros, inputs)
        works_pad = nn.functional.pad(works, pad=(2*hk, 2*hk, 2*hk, 2*hk) )
        #print("works_pad:",works_pad.shape)
        map_augs = []

        height_slice = works_pad.size(2) - 2 * hk  #获取填充后张量切片的大小
        width_slice = works_pad.size(3) - 2 * hk

        for i in range(ksize):
            for j in range(ksize):
                curr_in = works_pad[:, :, i:i+height_slice, j:j+width_slice]
                map_augs.append(curr_in)

        num_map = len(map_augs) # ksize*ksize
        center_map = map_augs[num_map//2]
        #print("center_map:",center_map.shape)
        #print("map_augs[0]:",map_augs[0].shape)
        peak_mask = center_map>map_augs[0]
        for n in range(1, num_map):
            if n == num_map // 2:
                continue
            peak_mask = peak_mask & (center_map>map_augs[n])
        
        peak_mask = peak_mask[:, :, hk:hk+height, hk:hk+width]
        
        if peak_mask.dtype != dtype:
            peak_mask = peak_mask.to(dtype)

        peak_mask.reshape(inputs.shape) # keep shape information

        return peak_mask

    def make_top_k_sparse_tensor(self,heatmaps, k=80, get_kpts=True):
        batch_size = heatmaps.shape[0]
        height = heatmaps.shape[2]
        width = heatmaps.shape[3]
        heatmaps_flt = heatmaps.reshape(batch_size,-1) 
        imsize = heatmaps_flt.shape[1]

        values, xy_indices = torch.topk(heatmaps_flt, k=k, dim=1, largest=True, sorted=False)
        boffset = torch.arange(batch_size, device=heatmaps_flt.device).view(-1, 1) * imsize
        indices = xy_indices + boffset
        indices = indices.view(-1)
        top_k_maps = torch.zeros(batch_size * imsize, device=heatmaps_flt.device)
        top_k_maps.scatter_(0, indices, 1)
        top_k_maps = top_k_maps.view(batch_size, 1 , height, width)
        if get_kpts:
            kpx = xy_indices % width
            kpy = xy_indices // width
            batch_inds = torch.arange(batch_size, dtype=torch.int32, device=heatmaps_flt.device).repeat_interleave(k)
            kpts = torch.cat([kpx.view(-1, 1), kpy.view(-1, 1)], dim=1)  # B*K, 2
            num_kpts = torch.full([batch_size], k, dtype=torch.int32, device=heatmaps_flt.device)
            return top_k_maps, kpts, batch_inds, num_kpts
        else:
            return top_k_maps    

    def build_multi_scale_deep_detector_3DNMS(self,input):
        photos=self.instance_normalization(input)

        batch_size = photos.shape[0]
        height = photos.shape[2]
        width = photos.shape[3]

        score_maps_list, det_endpoints = self.get_model(photos)
        scale_factors = det_endpoints['scale_factors']
        scale_factors_tensor = torch.from_numpy(scale_factors).float()
        num_scale = len(score_maps_list)
        scale_logits = [None] * num_scale

        for i in range(num_scale):
            logits = self.instance_normalization(score_maps_list[i])
            logits = nn.functional.interpolate(logits, size=(height, width), mode='bilinear', align_corners=False) # back to original resolution
            scale_logits[i] = logits
        scale_logits = torch.cat(scale_logits, dim=1) # [B,S,H,W]

        scale_heatmaps = self.soft_nms_3d(scale_logits, ksize=15, com_strength=3.0)

        max_heatmaps, max_scales = self.soft_max_and_argmax_1d(scale_heatmaps, dim=1, 
                                                    inputs_index=scale_factors_tensor)
        max_heatmaps = max_heatmaps.unsqueeze(1) # make max_heatmaps the correct shape
        #tf.summary.histogram('max_scales', max_scales) 画直方图

        eof_masks_pad = self.end_of_frame_masks(height, width, det_endpoints['pad_size'])


        max_heatmaps = max_heatmaps * eof_masks_pad
        # Extract Top-K keypoints
        eof_masks_crop = self.end_of_frame_masks(height, width, 16)

        nms_maps = self.non_max_suppression(max_heatmaps, 0.0, 5)
        nms_scores = max_heatmaps * nms_maps * eof_masks_crop
        top_ks,kpts,_,_ = self.make_top_k_sparse_tensor(nms_scores, k=80)
        top_ks = top_ks * nms_maps
        top_ks = top_ks.detach()  # 这里的 your_tensor 将不会计算梯度

        return top_ks

    def build_from_features(self,images):
        #input = torch.from_numpy(input)
        #curr_in=input[None,None,...].float()
        #input=input.squeeze(1)
        #assert input.dim()==4
        #print("images:",images.shape)
        if(images.dim()==5 and images.shape[1]!=1):
            split_input=torch.split(images,1,dim=1)
        elif(images.dim()==5 and images.shape[1]==1):
            split_input=[images.squeeze(1)]
        else:
            split_input=[images]

        stack_input=[]

        for input in split_input:
            if(input.dim()==5):
                input=input.squeeze(1)
            curr_in=self.instance_normalization(input)
            batch_size = curr_in.shape[0]
            height = curr_in.shape[2]
            width = curr_in.shape[3]

            scale_heatmaps = self.soft_nms_3d(curr_in, ksize=15, com_strength=3.0)

            max_heatmaps, _ = self.soft_max_and_argmax_1d(scale_heatmaps, dim=1)

            max_heatmaps = max_heatmaps.unsqueeze(1) # make max_heatmaps the correct shape
            #tf.summary.histogram('max_scales', max_scales) 画直方图

            pad_size=4
            eof_masks_pad = self.end_of_frame_masks(height, width, 4)
            #原式为endpoints['pad_size'] = num_conv * (conv_ksize//2) 
            #经过肉眼观察，dpvo中提取特征图卷积次数为4，ksize为3，故padsize应为4
            eof_masks_pad=eof_masks_pad.to(device="cuda")
            max_heatmaps = max_heatmaps * eof_masks_pad
            # Extract Top-K keypoints
            eof_masks_crop = self.end_of_frame_masks(height, width, 16)
            eof_masks_crop=eof_masks_crop.to(device="cuda")
            #print("eof_masks_crop:",eof_masks_crop.shape)
            nms_maps = self.non_max_suppression(max_heatmaps, 0.0, 5)
            nms_scores = max_heatmaps * nms_maps * eof_masks_crop
            top_ks,kpts,_,_ = self.make_top_k_sparse_tensor(nms_scores, k=48) #这里调补丁数的
            top_ks = top_ks * nms_maps
            top_ks = top_ks.detach()  # 这里的 your_tensor 将不会计算梯度
            #print(kpts.shape)

            stack_input.append(kpts)
            
    
        stacked_top_ks=torch.stack(stack_input,dim=0)

        
        return stacked_top_ks

def calculate_padding(kernel_size, stride, input_size):
        pad = max((input_size - 1) * stride + kernel_size - input_size, 0)
        pad_left = pad // 2
        pad_right = pad - pad_left
        return pad_left, pad_right

if __name__ == '__main__':
    image = cv2.imread("1.png")   
    if image is None:
        raise ValueError("Image not found or unable to read.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

    #转化为灰度图
    if image.ndim == 3 and image.shape[-1] == 3:
        input = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        input=image

    #转化为NSHW
    input = torch.from_numpy(input)
    curr_in=input[None,None,...].float()
    assert curr_in.dim()==4
    curr_in=curr_in[None,...]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lf_test=LF_Net(curr_in).to(device)
    curr_in=curr_in.to(device)
    lf_test(curr_in)

    
