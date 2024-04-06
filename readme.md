it is a copy of orginal 3d gaussian splatting source code
used for learning

# Parameter Track
## gaussian created:
* _feature_dc [100000,1,3]
* _feature_rest [100000,15,3]
* _opacity [100000,1]
* _rotation [100000,4]
* _scaling [100000,3]
* _xyz [100000,3]

## send to rasterizer
* means3D = pc.get_xyz [100000,3]
* means2D = screenspace_points [100000,3]
* shs = pc.get_features [100000,16,3]
* opacities = pc.get_opacity [100000,1]
* scales = pc.get_scaling [100000,3]
* rotations = pc.get_rotation [100000,4]


## backward
我们已经有：
* dL_dout_color：loss对forward输出颜色的梯度{1920x1080，3} （其中loss为GT图像和output图像的 L1+SSIM）

我们保存了以下tensor可能被用于backward计算梯度
* colors_precomp: 我们没有提供，可以忽略这个
* means3D：pc.get_xyz [100000,3] 世界坐标系下高斯中心坐标
* scales = pc.get_scaling [100000,3]
* rotations = pc.get_rotation [100000,4]
* radii：forward的返回结果，屏幕坐标下高斯投影近似圆的半径
* sh = pc.get_features [100000,16,3] 球谐系数
* geomBuffer 超大块连续显存的起始地址
* binningbuffer 超大块连续显存的起始地址
* imgbuffer 超大块连续显存的起始地址
```C++
    struct GeometryState
	{
		size_t scan_size;
		float* depths; //@@ preprocess
		char* scanning_space;
		bool* clamped; //@@ preprocess
		int* internal_radii;
		float2* means2D; //@@ preprocess
		float* cov3D; //@@ preprocess
		float4* conic_opacity; //@@ preprocess
		float* rgb; //@@ preprocess
		uint32_t* point_offsets; //@@ 前缀和
		uint32_t* tiles_touched; //@@ preprocess

		static GeometryState fromChunk(char*& chunk, size_t P);
	};
    struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};
```

我们需要最终计算出下面几项梯度：
* dL_dmeans2D：loss对于投影之后世界坐标系高斯中心点的梯度 {100000, 3}
* dL_dopacity：loss对于高斯中心不透明度的梯度
* dL_dmeans3D：loss对于世界坐标系下高斯中心坐标的梯度 {100000, 1}
* dL_dsh loss对每个球谐系数的梯度 {100000, 16, 3}
* dL_dscales：{100000, 3}
* dL_drotations：{100000, 3}
* 没提供，不用算 dL_dcolors：loss对于预先计算高斯颜色的梯度 {100000, 3}，
* 没提供，不用算 dL_dcov3D：loss对预先计算的协方差矩阵的地图 {100000, 3}，

### BACKWARD::render
我们在forward中执行的最后一个函数是render，我们backward中第一执行的就是render。  
在backward::render中，我们可以求出最终输出偏微分dL_dmean2D和dL_dopacity，以及中间变量偏微分dL_dconic， dL_dcolor

* const float dL_dpixel[3]：当前像素L对RGB 3通道值的偏微分 已有
* const float ddelx_dx,ddely_dy : 屏幕空间坐标对NDC坐标的偏微分 已有
* d = f(G_xy, P_xy) = { xy.x - pixf.x, xy.y - pixf.y };
* power = f(con_o.x, con_o.y, con_o.z, d) = = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
* G = exp(power)
* alpha = f(con_o.w, G) =  min(0.99f, con_o.w * G);

### BACKWARD::preprocess
第二部执行BACKWARD::preprocess，我们从backward::render中得到了一些中间变量偏微分 dL_dmean2D，dL_dconic2D,dL_dcolor。我们会用他们来计算 dL_dmean3D,dL_dsh，dL_dscale,dL_drot。
其中dL_dscale,dL_drot都是通过backward_preprocess中的dL_dcov3d这个中间变量得出。
