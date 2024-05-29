# Readme

代码部分是带我注释的原始版 3d gaussian splatting 论文代码。  
至少应该通读代码并运行调试一遍，以加深对以高斯为基元的三维表示的理解。
如何通读？从train.py的main函数开始按照运行顺序逐行阅读，重点关注和学习以下几点：

* 高斯属性存储的数据结构，每个属性的定义，作用。
* pytorch调用自定义cuda模块的流程方法。
* cuda前向代码，这是学习cuda编程的很好的案例，应该深入理解。

## 3DGS notes
### torch.nn.Module's subclass
当我们创建一个继承自torch.nn.module的类时，我们通常需要定义forward方法来描述前向传播的过程。反向传播的计算是通过自动微分来实现的，因此我们不需要显示的定义backward反法，pytorch会自动记录前向传播中的操作自动计算反向传播。
### Parameter Track

#### gaussian created:

* _feature_dc [100000,1,3]
* _feature_rest [100000,15,3]
* _opacity [100000,1]
* _rotation [100000,4]
* _scaling [100000,3]
* _xyz [100000,3]

#### send to rasterizer

* means3D = pc.get_xyz [100000,3]
* means2D = screenspace_points [100000,3]
* shs = pc.get_features [100000,16,3]
* opacities = pc.get_opacity [100000,1]
* scales = pc.get_scaling [100000,3]
* rotations = pc.get_rotation [100000,4]


#### backward

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

##### BACKWARD::render

我们在forward中执行的最后一个函数是render，我们backward中第一执行的就是render。  
在backward::render中，我们可以求出最终输出偏微分dL_dmean2D和dL_dopacity，以及中间变量偏微分dL_dconic， dL_dcolor

* const float dL_dpixel[3]：当前像素L对RGB 3通道值的偏微分 已有
* const float ddelx_dx,ddely_dy : 屏幕空间坐标对NDC坐标的偏微分 已有
* d = f(G_xy, P_xy) = { xy.x - pixf.x, xy.y - pixf.y };
* power = f(con_o.x, con_o.y, con_o.z, d) = = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
* G = exp(power)
* alpha = f(con_o.w, G) =  min(0.99f, con_o.w * G);

##### BACKWARD::preprocess

第二部执行BACKWARD::preprocess，我们从backward::render中得到了一些中间变量偏微分 dL_dmean2D，dL_dconic2D,dL_dcolor。我们会用他们来计算 dL_dmean3D,dL_dsh，dL_dscale,dL_drot。
其中dL_dscale,dL_drot都是通过backward_preprocess中的dL_dcov3d这个中间变量得出。

## 2DGS

project page：https://surfsplatting.github.io/

### Paper

最近3DGS的出现颠覆了辐射场重建领域，它达到了高质量的新视角合成以及高速渲染的目的。但是由于三维椭球基元的多视角不一致性，3DGS无法很好的精准表示物体表面。我们提出2D Gaussian splatting，这是一个从多视角图片建模并重建几何精准的辐射场的方法。我们的核心idea是把三维体积坍缩成一组2D的平面高斯圆盘。 不同于3D高斯，2D高斯在对表面进行建模的同时提供了视角一致性的几何形状。为了能够精准的恢复出薄薄的物体表面并且达到稳定的优化，我们引入了利用射线-splat相交和光栅化的透视精确的 2D splatting过程。此外，我们还结合了depth distortion(深度畸变)和normal consistency(法线一致)loss项，以进一步提高重建的质量。我们证明，我们的可微渲染器允许无噪声和详细的几何重建，同时保持有竞争力的外观质量、快速训练速度和实时渲染。

#### 1. Intro

逼真的新颖视图合成（NVS）和精确的几何重建是计算机图形和视觉领域的关键长期目标。 最近，由于3DGS的实时渲染和照片级真实度的NVS结果，3D高斯泼溅已成为隐式(NeRF)和基于网格的特征表示(Instant-NGP)的一个有吸引力的替代方案。3DGS发展迅速，已快速扩展到多个领域，包括抗锯齿渲染，材质建模，动态场景重建，可动数字人。然而，它在捕捉复杂的几何形状方面存在不足，因为对完整角辐射进行建模的有体积 3D高斯与真实物体表面的薄特性相冲突。  

在另一方面，一篇早期工作[Pfister et al. 2000; Zwicker et al.2001a]已经展示了surfels是一种能够表示复杂几何的高效基元。Surfels在一个局部近似物体表面并带有形状和着色属性，并且可以从已知几何中提取出来。surfels被广泛的运用在SLAM和其他的一些机器人任务中作为一种高效的几何表示。随后的进展将 surfels 纳入了可微分框架。不过，这些方法通常需要groud truth几何图形、深度传感器数据，或在已知照明的受限场景下运行。

受到上面这些工作的启发，我们提出了2DGS用于三维场景的重建和新视角合成，它同时结合了两边的优点，又解决了他们的局限性。与3DGS不同，我们的方法用2D高斯基元来表示一个三维场景。2D高斯的最重要的有点就是他在渲染过程中精确的几何表示。特别的，3DGS评估那些像素光线相交的高斯的值，这在不同视角渲染下会导致深度不连续。相比之下，我们的方法利用了显示的光线-泼溅相交算法，会产生透视精准的泼溅结果，如图2所示，这直接提高了重建精度。而且2D高斯基元本身的表面发现可以直接通过进行法线约束来进行表面正则化。对比其他基于片元模型，我们的2D高斯可以从未知几何状态的数据中，通过基于梯度的优化恢复出几何。  

尽管我们的2D高斯方法在几何建模上表现优秀，但是仅仅使用photometric loss进行优化会导致重建过程中出现噪声。为了增强重建效果并且获得更加平滑的表面，我们提出了两个正则化项，分别是depth distortion和normal consistency。depth distortion会聚集2D高斯基元在光线方向上分布在一个很小的区间里，解决了渲染过程中高斯之间的距离被忽略的限制。normal consistency最小化了渲染的法线图和深度图梯度之间的差异，确保深度和法线定义的几何之间对齐。用这两个正则化项能够帮助我们提取更高精度的表面mesh。本文主要又以下贡献：

* 我们给出了高效的可微2D高斯渲染器。
* 提出了两个正则化项
* 我们的方法达到了SOTA级别的重建精度和NVS结果。

#### 2. related work

skipped

#### 3. 3DGS

参考3DGS原文  

3DGS在表面重建中面临的挑战：

* 3DGS的体积表示和物体表面是薄层的特性相冲突。
* 3DGS本身不对物体表面法线建模，法线对高质量模型很重要。
* 3DGS的光栅化流程缺乏多视图一致性，不同视点会产生不一样的2D投影平面。
* 3DGS使用仿射矩阵将三维高斯转换到射线空间只能获得中心附近的精确投影，而周围区域的透视精度则会大打折扣。

#### 4. 2DGS

##### 4.1 modeling

与3DGS不同，它对斑点中的整个角辐射进行建模，我们通过采用嵌入 3D空间中的“平坦”2D高斯函数来简化三维建模。通过二维高斯建模，2D高斯基元将密度分布在平面圆盘内，将法线定义为密度变化最陡的方向。这些特性使得2D高斯基元能更好的对齐薄薄的表面。虽然以前的工作也使用2D高斯来做几何重建，但是他们需要稠密点云和法线真值作为输入。相比之下，我们仅在稀疏校准点云和光度监督的情况下同时重建外观和几何形状。  

如图3所示，我们的2D泼溅有一个中心点p_k，两个主切线相邻t_u和t_v，还有一个缩放因子向量 S = (S_u, S_v) 来控制2D高斯变化。法线t_w则被两个正交切线向量叉乘得到。我们定义一个2D高斯基元的朝向为一个3x3的旋转矩阵 R = [t_u, t_v, t_w]。而缩放因子则被表示为一个3x3的对角阵，且对角最后一位为0。  

一个2D高斯因此可以被定义为一个世界空间下的一个局部切平面，可以被参数化为：...skipped

对于uv坐标系下的一个点(u，v)，可以根据高斯分布公式算出这个点的值。  

中心点p_k， 缩放因子向量 S = (S_u, S_v)，和旋转矩阵 R = [t_u, t_v, t_w]为可学习参数。和3DGS一样，每一个2D高斯基元也有有不透明度和球谐系数。

##### 4.2 splatting

渲染2D高斯的一种常见策略是使用透视投影的仿射近似将2D高斯基元投影到图像空间上(像3DGS一样)，但是正如之前提到的，这个投影只在高斯中心是精确的，离高斯中心越远的地方误差越大。为了解决这个问题，Zwicker提出了一种基于齐次坐标系的公式。特别的，2D泼溅投影到一个图像平面上可以被描述成一个普通的2D到2D齐次坐标系下的映射。令4x4的W矩阵为世界空间到屏幕空间的变换矩阵。屏幕空间下点的坐标可以被下式得到。

* x = (𝑥𝑧, 𝑦𝑧, 𝑧, 𝑧)^⊤ = W𝑃 (𝑢, 𝑣) = WH(𝑢, 𝑣, 1, 1)^⊤  

其中x代表了一条从相机触发经过像素(𝑥，𝑦)的光线，在深度z与一个splat相交。为了光栅化一个2D高斯，Zwicker提出了使用M = (WH)^（−1）的隐式方法，将椭圆的圆锥曲线投影到屏幕空间。然而逆变换会带来数值上的不确定性，尤其当椭圆片退化成一条线的时候(从侧面观看)。为了解决这个问题，以前的表面泼溅渲染方法使用预定义的阈值丢弃这种病态变换。然而，这种方案在可微分渲染框架内提出了挑战，因为阈值化可能导致优化不稳定，为了解决这个问题，我们使用的显式的光线splat相交。  

**Ray-splat Intersection:** 通过找到三个不平行平面的交点，我们可以有效地找到射线平分线的交点。这种方法原本是为了一个特殊的硬件而设计的。给定一个图片上的坐标x = (𝑥, 𝑦)，我们将像素的光线参数化为两个正交平面的交集：x平面和y平面。x平面被一个法向量(-1，0，0)和一个偏移量x所定义。因此，x平面可以被表示为4d齐次平面hx=(-1，0，0，x)，类似的y平面可以被表示为hy=(0,-1,0,y).因此光线x=(x,y)就被x平面和y平面的相交线所决定。  
下一步，我们把两个平面变换到2D高斯基元的本地坐标(uv坐标)，注意以下把平面上的一个点用变换矩阵M进行变换等价于用M的逆转置矩阵对齐次平面参数进行变换。...


#### training

我们的2D高斯方法，能够有效的进行几何建模，但是如果只使用photometric loss会导致有噪声的重建，这是3D重建任务本身所导致的，为了解决这个问题并提升几何重建，我们提出了两种正则化项、

##### Depth Distortion
与NeRF，3DGS不同，他们的体素渲染不考虑两个高斯基元之间的距离。因此分散的高斯基元最后可能也会得到相似的颜色和深度渲染。这和表面渲染不同，光线只穿过可见的表面一次。为了解决这个问题，我们受到MipNerf360的启发，并使用了深度畸变loss来让高斯基元再光线方向上的分布更为集中，通过最小化ray-splat相交之间的距离。
