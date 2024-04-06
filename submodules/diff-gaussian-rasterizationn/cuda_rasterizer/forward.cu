/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	//@@ 高斯点的世界坐标
	glm::vec3 pos = means[idx];
	//@@ 观察方向：归一化（高斯点坐标-相机坐标）
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	//@@ 读取球谐系数
	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	//@@ 具体的回复过程很复杂，以后再研究
	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	//@@ 如果RGB算出来有负的，需要把他变成0，并记录一下，后面反向要用
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	
	//@@ 将世界坐标系点转换到视图空间
	float3 t = transformPoint4x3(mean, viewmatrix);

	//@@ 如果视图空间下该点在FOV之外，把他们拉回FOV之内
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	//@@ 构建雅可比矩阵
	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	//@@ W是world2view矩阵的3*3左上角
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	//@@ 从6个参数回复cov3D
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	//@@ JW SIGMA WJ
	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;

	//@@ 返回cov2D的三个参数
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	//@@ 创建scale矩阵，对角阵，对角线是三轴scale值
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	//@@ 标准化四元数，rot(w,x,y,z),把每个值单独拿出来，好计算
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	//@@ 从四元数得到旋转矩阵
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	//@@ S*R
	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	//@@ 这就是sigma = RSSR的那个论文公式
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	//@@ cov是个对角阵，只存上三角六个参数
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	//@@ 计算当前thread的id
	auto idx = cg::this_grid().thread_rank(); 
	//@@ 如果当前thread的id超过了高斯数量，该thread不做任何事情
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	//@@ 初始化thead处理的高斯的radii和tiles_touched为0，如果这两个值没变，那么这个高斯就不会再被处理。
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	//@@ 近裁剪，如果这个thread处理的高斯在视锥以外，该thread不做任何事，
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting 投影
	//@@ 找到该高斯在世界坐标系下的坐标
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	//@@ 经过全投影，将世界坐标系下的高斯坐标转换到投影空间，是个float4 [x,y,depth,w]
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	//@@ 计算缩放因子，加上一个很小的数，防止分母为0
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	//@@ 转换回float3， 方便后续使用
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w }; 

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	//@@ 如果提供了事先计算的cov，直接用，否则自己算
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		//@@ 用S和R计算当前高斯的cov，并赋值给cov3Ds(geobuffer)
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		//@@ 从geobuffer中的cov3Ds读取cov3D
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	//@@ 计算屏幕空间中的cov2D
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	//@@ cov2D的行列式，如果行列式为0，该矩阵不可逆，thread中止
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	//@@ conic是cov2D的逆矩阵
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles.
	//@@ 计算cov2D的特征值
	//@@ 在二维空间中，二维高斯分布的等概率轮廓可以用以特征向量为轴的椭圆表示，其中椭圆的长轴和短轴的长度由特征值的平方根决定。
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	//@@ 用3个标准差以内的长轴长度当作半径，三个标准差覆盖了高斯99%的概率，多余的可以忽略不记
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	//@@ 把归一化设备坐标(NDC)转化成屏幕像素坐标
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	//@@ rectmin和rectmax是当前椭圆的近似圆所覆盖的多个tile区域矩形的左下角编号和右上角编号
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	//@@ 如果有任何一个方向没有覆盖任何tile，中止该thread
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	//@@ SH转换成RGB，并且存放到geobuffer里
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	//@@ 深度，半径，像素坐标都存一下
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	//@@ 把cov2D的逆和不透明度揉在一起 存一下
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	//@@ tiles_touched是一个数，覆盖到的tile的数量
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges, //@@ tile的高斯起止范围
	const uint32_t* __restrict__ point_list, //@@ 有序有效的高斯列表
	int W, int H, //@@ 图片宽高
	const float2* __restrict__ points_xy_image, //@@ 高斯点像素坐标
	const float* __restrict__ features, //@@ 高斯rgb
	const float4* __restrict__ conic_opacity, //@@ cov2D和不透明度
	float* __restrict__ final_T, //@@ 最终透过率
	uint32_t* __restrict__ n_contrib, //@@ 贡献者数量
	const float* __restrict__ bg_color, //@@ 背景颜色
	float* __restrict__ out_color //@@ 最终返回颜色)
{
	// Identify current tile and associated min/max pixel range.
	//@@ 获取当前block id也就是tile id
	auto block = cg::this_thread_block();
	//@@ 横向的block数量，或者说一张图x方向上铺了多少个tile
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	//@@ 获取当前tile的左下角像素和右上角像素坐标
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	//@@ 获取当前像素坐标
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	//@@ 计算pix的thread ID
	uint32_t pix_id = W * pix.y + pix.x;
	//@@ 类型转换成float
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	//@@ 判断这是否是一个有效像素
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	//@@ 如果超出图片范围则标记为完成
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	//@@ 加载当前tile的覆盖高斯范围
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	//@@ 如果这个tile覆盖的高斯太多，多到超过了一个block，就要多进行一轮cuda并行运算
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x; //@@ 需要处理的高斯数量

	// Allocate storage for batches of collectively fetched data.
	//@@ 开辟共享内存，保存高斯id，高斯像素坐标，高斯cov2D和opacity
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f; //@@ 透过率累积
	uint32_t contributor = 0; //@@ 颜色贡献者数量
	uint32_t last_contributor = 0; //@@ 上一个颜色贡献者id
	float C[CHANNELS] = { 0 }; //@@ rgb通道

	// Iterate over batches until all done or range is complete
	//@@ 循环处理高斯分布的每一个批次，每完成一个批次，todo就减去一个blocksize
	//@@ 由于一个block就是一个tile，即block里的每一个像素交的高斯列表是完全相同的
	//@@ 这里的一个批次中，每个像素最多循环处理256个高斯，最少循环处理todo个高斯，
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		//@@ 检测当前block中所有thread 的 done变量情况。
		int num_done = __syncthreads_count(done);
		//@@ 如果都处理完毕了，循环结束。
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		//@@ 每个批次中，每个thread首先找一个高斯把它放进共享内存
		int progress = i * BLOCK_SIZE + block.thread_rank(); //@@ 获取当前thread正在处理的高斯在所有批次中的编号
		if (range.x + progress < range.y) //@@ 只有progress还在范围内的时才继续执行
		{
			int coll_id = point_list[range.x + progress]; //@@ 从有序高斯表中找到这个高斯的全局id
			collected_id[block.thread_rank()] = coll_id;  //@@ 存入这个高斯的所有信息到共享内存中，方便当前批次所有线程访问
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		//@@ 线程同步点
		block.sync();

		// Iterate over current batch
		//@@ 开始遍历批次内的所有高斯
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			//@@ 贡献者++，也代表了这是在处理从前往后第几个贡献者
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j]; //@@ 获取高斯中心的屏幕坐标
			float2 d = { xy.x - pixf.x, xy.y - pixf.y }; //@@ 计算当前像素的屏幕坐标到高斯屏幕坐标的距离
			float4 con_o = collected_conic_opacity[j]; //@@ 获取高斯cov2d和opcaity
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y; //@@ 计算一个power，代表这个像素受到该高斯影响的程度大小
			if (power > 0.0f) //@@ 如果power是个正值，说明这个高斯对本像素的影响小到可以忽略不记，取处理下一个高斯
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power)); //@@ 计算本像素在当前高斯影响下的不透明度
			if (alpha < 1.0f / 255.0f) //@@ 如果不透明度过小，忽略不计
				continue;
			float test_T = T * (1 - alpha); //@@ 更新透过率
			if (test_T < 0.0001f) //@@ 如果透过率很低了，说明这个像素颜色叠加的差不多了，标记为done。该像素处理完毕了。
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			//@@ 给本像素的rgb通道填充颜色
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	//@@ 处理最终输出
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{	
	//@@ 启动核函数
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	//@@ 启动核函数，启动单位是每个高斯点
	preprocessCUDA<NUM_CHANNELS> <<<(P + 255) / 256, 256 >>> (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}