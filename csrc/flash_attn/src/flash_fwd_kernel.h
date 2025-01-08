/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "philox_unpack.cuh" // For at::cuda::philox::unpack

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "dropout.h"
#include "rotary.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, \
        bool Is_softcap, bool Split, bool Append_KV, typename Params>
inline __device__ void compute_attn_1rowblock_splitkv(const Params &params, const int bidb, const int bidh, \
                                                const int m_block, const int n_split_idx, const int num_n_splits) {
    
    // break flash_fwd_kernel.h:686 if blockIdx.y==3 && blockIdx.z==7

    // const int m_block = blockIdx.x;

    /* (cuda-gdb) p *(@generic Flash_fwd_params*)&params
    $33 = {__b_10Qkv_params = {q_ptr = 0x7ffe8fe00000, k_ptr = 0x7ffeca600000, v_ptr = 0x7ffecaba0000, 
        q_batch_stride = 768, k_batch_stride = 196608, v_batch_stride = 196608, q_row_stride = 768, 
        k_row_stride = 768, v_row_stride = 768, q_head_stride = 64, k_head_stride = 64, v_head_stride = 64, 
        h = 12, h_k = 12, h_h_k_ratio = 1}, o_ptr = 0x7ffe8ffea600, oaccum_ptr = 0x0, o_batch_stride = 768, 
      o_row_stride = 768, o_head_stride = 64, p_ptr = 0x0, softmax_lse_ptr = 0x7ffe8fe02600, 
      softmax_lseaccum_ptr = 0x0, b = 5, seqlen_q = 1, seqlen_k = 768, seqlen_knew = 0, d = 64, 
      seqlen_q_rounded = 128, seqlen_k_rounded = 768, d_rounded = 64, rotary_dim = 0, total_q = 0, 
      scale_softmax = 0.125, scale_softmax_log2 = 0.180336878, cu_seqlens_q = 0x0, cu_seqlens_k = 0x7ffe8fe02000, 
      leftpad_k = 0x0, seqused_k = 0x0, blockmask = 0x0, knew_ptr = 0x0, vnew_ptr = 0x0, knew_batch_stride = 0, 
      vnew_batch_stride = 0, knew_row_stride = 0, vnew_row_stride = 0, knew_head_stride = 0, 
      vnew_head_stride = 0, rotary_cos_ptr = 0x0, rotary_sin_ptr = 0x0, cache_batch_idx = 0x0, 
      block_table = 0x7ffe8fe01e00, block_table_batch_stride = 3, page_block_size = 256, p_dropout = 1, 
      p_dropout_in_uint8_t = 255 '\377', rp_dropout = 1, scale_softmax_rp_dropout = 0.125, 
      window_size_left = 768, window_size_right = 0, softcap = 0, philox_args = {seed_ = {val = 0, ptr = 0x0}, 
        offset_ = {val = 0, ptr = 0x0}, offset_intragraph_ = 0, captured_ = false}, rng_state = 0x0, 
      is_bf16 = false, is_causal = true, is_seqlens_k_cumulative = false, is_rotary_interleaved = false, 
      num_splits = 1, alibi_slopes_ptr = 0x7ffe8fe02c00, alibi_slopes_batch_stride = 12, unpadded_lse = false, 
      seqlenq_ngroups_swapped = false} */

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;   // 64   这两个值在 run_mha_fwd_splitkv_dispatch 编译期间固化
    constexpr int kBlockN = Kernel_traits::kBlockN;   // 256
    constexpr int kHeadDim = Kernel_traits::kHeadDim; // 64
    constexpr int kNWarps = Kernel_traits::kNWarps;   // 4

    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb/*0*/);
    // binfo = {sum_s_q = -1, sum_s_k = -1, actual_seqlen_q = 1, leftpad_k = 0, seqlen_k_cache = 124, actual_seqlen_k = 124}

    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("Is_even_MN = %d, is_cumulativ = %d, seqlen_k_cache = %d, actual_seqlen_k = %d\n", Is_even_MN, params.is_seqlens_k_cumulative, binfo.seqlen_k_cache, binfo.actual_seqlen_k); }
    // if (threadIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("params.knew_ptr = %p, seqlen_k_cache + seqlen_knew = %d\n", params.knew_ptr, binfo.seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)); }
    if (m_block * kBlockM/*64*/ >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split /*1*/ = ((params.seqlen_k/*128*/ + kBlockN - 1) / kBlockN + num_n_splits/*1*/ - 1) / num_n_splits;
    const int n_block_min/*0*/ = !Is_local // false
        ? n_split_idx * n_blocks_per_split
        : std::max(n_split_idx * n_blocks_per_split, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max/*1*/ = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);
    // 如果k v 很长的, 如300, 那么n_block_max = 2, 每256个一个block.
    if (Is_causal /*true*/ || Is_local /*false*/) {
        n_block_max = std::min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - \
                                                binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }
    
    if (true && threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        print("\nn_block_min: "); print(n_block_min);
        print("\nn_block_max: "); print(n_block_max);
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    // We move K and V to the last block.
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = params.block_table == nullptr ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const int block_table_idx = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN / params.page_block_size;
    const int block_table_offset = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN - block_table_idx * params.page_block_size;
    const index_t row_offset_k = block_table == nullptr
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : block_table[block_table_idx] * params.k_batch_stride + block_table_offset * params.k_row_stride + \
        (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = block_table == nullptr
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : block_table[block_table_idx] * params.v_batch_stride + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + \
                                            binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    // 这个gQ是一个切出来的tile
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM/*64*/>, Int<kHeadDim/*64*/>>{},
                           make_coord(m_block/*0*/, 0));  // (kBlockM, kHeadDim)
    if (false && threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        print("\nmQ's content: "); print_tensor(mQ);
        print("\ngQ's content: "); print_tensor(gQ);
    }
    // gK gV不是切出来的tile, 只是一个指定尺寸的tensor
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN/*256*/>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("k_ptr = %p, row_offset_k = %d, gK_ptr = %p\n", params.k_ptr, row_offset_k, gK.data()); }
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));

    // shared memory里的 q k v vt(转置的v), 这里只是预留shared memory的空间, 此时还没有存放数据
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    // 以上sQ sK sV sVt 都是在shared memory的, 所有默认是带有swizzle的, 
    // 下面的sVtNoSwizzle是sV的转置, 但是没有swizzle, 用来计算softmax
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        print("\nGlobal & shared memory data: ");
        print("\nmQ: "); print(mQ);     //print_tensor(mQ);
        print("\ngQ: "); print(gQ);     //print_tensor(gQ);
        print("\ngK: "); print(gK);
        print("\ngV: "); print(gV);
        print("\nsQ: "); print(sQ);
        print("\nsK: "); print(sK);
        print("\nsV: "); print(sV);
        print("\nsVt: "); print(sVt);
        print("\nsVtNoSwizzle: "); print(sVtNoSwizzle);
        print("\n");
    }
    // 1. 这个 gmem_tiled_copy_QKV 以及 gmem_thr_copy_QKV是用来 copy 数据从global memory -> shared memory 的
    // 2. 这个tiled跟后边的tile_mma的tile不是一个tile, 这里的tile仅仅是一块一块的copy的意思, tile的大小跟后边tiled_mma
    // 的大小也没关系.
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    // 分别是全局内存&shared memory的 q k v, 被partition成线程级别的tensor
    // S D分别是source destination的意思, 就是接触tiledcopy把数据从S cp到D
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma; // 这个变量用来控制tile的mma
    typename Kernel_traits::TiledMma2 tiled_mma2; // 这个变量用来控制tile的mma
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    auto thr_mma2 = tiled_mma2.get_thread_slice(tidx);
    // 分别是线程级别的 存在寄存器里的要即将做mma的tensor
    // 意思是获取线程级别的 矩阵乘法的A B数据, 也就是 Q K数据, A B数据的编排是不一样的, 有点类似tensor core的数据编排
    // 所以有了A B的区分, 可以参考 ptx mma.m16n8k16的thread value编排.
    // AB计算后的结果C, 也有线程value编排, 可以见 ptx的mma.m16n8k16.
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrQ2  = thr_mma2.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tSrK2  = thr_mma2.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    if (false && threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        printf("\n=======TILED_COPY========");
        print("\ngmem_thr_copy_QKV: "); print(gmem_thr_copy_QKV);
        print("\ntQgQ: "); print(tQgQ); //print_tensor(tQgQ);
        print("\ntQsQ: "); print(tQsQ); //print_tensor(tQsQ);
        print("\ntKgK: "); print(tKgK);
        print("\ntKsK: "); print(tKsK);
        print("\ntVgV: "); print(tVgV);
        print("\ntVsV: "); print(tVsV);
    }

    if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        printf("\n=======TILED_MMA========");
        print("\ntiled_mma:"); print(size(tiled_mma));
        //print("\ntiled_mma2:"); print(tiled_mma2);
        print("\nthr_mma:"); print(thr_mma);
        print("\ntSrQ: "); print(tSrQ); //print_tensor(tSrQ);
        print("\ntSrQ2: "); print(tSrQ2); //print_tensor(tSrQ);
        print("\ntSrK: "); print(tSrK);
        print("\ntSrK2: "); print(tSrK2);
        print("\ntOrVt: "); print(tOrVt);
    }

#if 1

    //
    // Copy Atom retiling
    //

    // 这个smem_tiled_copy_Q 是按照tiled_mma, 把Q数据从shared memory cp 到register的
    // 这个copy和 tiled_mma是强绑定的, 因为copy后的数据要按照tiled_mma来计算.
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    if (false && threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        print("\ntSsQ: "); print(tSsQ); print(" stride: "); print(tSsQ.stride());
        print("\ntSsK: "); print(tSsK); print(" stride: "); print(tSsK.stride());
        print("\ntOsVt: "); print(tOsVt); print(" stride: "); print(tOsVt.stride());
    }
    // PREDICATES
    //

    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // Construct identity layout for sQ and sK
    // 构造只有形状没有类型的tensor，用于一些特定变换
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));


    // Prologue

    // Copy from Knew to K, optionally apply rotary embedding.

    // Read Q from gmem to smem, optionally apply rotary embedding.
    if (!Append_KV || params.rotary_dim == 0) {
        // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                           binfo.actual_seqlen_q - m_block * kBlockM);
    }

    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    // 这里是 global memory 拷贝到 shared memory, 异步的, 因此需要fence waitgroup等.
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV,
                                       binfo.actual_seqlen_k - n_block * kBlockN);
    // 在循环之前 提前cp q k.
    cute::cp_async_fence();

    // flash::cp_async_wait<0>();
    // __syncthreads();
    // if (tidx == 0 && blockIdx.y == 0 && blockIdx.z == 0) { print(tKsK); }
    // __syncthreads();

    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o)> softmax;

    if (false && threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        print("\nSoftmax_row_max: "); print(softmax.row_max);
        print("\nSoftmax_row_sum: "); print(softmax.row_sum);
    }

    const float alibi_slope = !Has_alibi ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    if (false && threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        print("\nn_masking_steps: "); print(n_masking_steps);
        print("\nn_block: "); print(n_block);
    }
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps/*2*/; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            print("\nacc_s: "); print(acc_s);
        }
        flash::cp_async_wait<0>();
        __syncthreads();

        // Advance gV
        if (masking_step > 0) {
        } else {
            // Clear the smem tiles to account for predicated off loads
            // 在gemm a k的前一时刻, 触发copy v
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        
        }
        cute::cp_async_fence();

        //void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
        //        Tensor4 const& tCsB, TiledMma tiled_mma,
        //        TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
        //        ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {

        // q k 计算
        // 这里的flash::gemm(也就是cute::gemm) 并不像cute的example那样 在for k_tile里, 有可能是因为flash-attn
        // 在并行度上已经非常大, 每个sm处理的就是类似 (1,64) (128,64) (128,64) 这样的小size的计算, 也就是说, 
        // 这里的k_tile数有可能就是1.
        flash::gemm(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // if (cute::thread0()) { print(acc_s); }


        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        flash::cp_async_wait<0>();
        __syncthreads();
        // if (tidx == 0 && blockIdx.y == 0 && blockIdx.z == 0) { print(tVsV); }
        // __syncthreads();

        // We have key_padding_mask so we'll need to Check_inf
        softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local || !Is_even_MN>(\
                                            acc_s, acc_o, params.scale_softmax_log2);
        // if (cute::thread0()) { print(scores_max); print(scores_sum); print(scores); }

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s); // 所有元素的exp序列, shape和 qk结果的shape一样
        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));

        // q k 计算的结果和v计算
        // 在gemm_rs(也就是qk结果和v计算)之前一刻, 会触发k的copy
        if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            print("\n**** flash_gemm_rs(): ");
            //print("\ntOrP: "); print(tOrP); // 这个print会导致后续数据不对, 校验失败, 进测试时打开
            print("\ntOrVt: "); print(tOrVt);
            print("\nacc_o: "); print(acc_o);
        }
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps/*2*/ > 1 && n_block/*0*/ <= n_block_min/*0*/) {
            --n_block;
            break;
        }
    }

    // Epilogue

    // 此时只是计算了最终的lse, 但是后边并没有使用.
    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(acc_o, params.scale_softmax);
    // if (cute::thread0()) { print(lse); }
    if (false && threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        print("\nlse: "); print(lse);
    }

    Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), \
                                                typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    using SmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::SmemCopyAtomO,
        typename Kernel_traits::SmemCopyAtomOaccum
    >;
    auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
    auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor rO = flash::convert_type<ElementO>(acc_o);
    Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
    if (false && threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        print("\n==========");
        print("\nsOaccum: "); print(sOaccum); // 64*64
        //print("\nrO: "); print(rO);
        //print("\ntaccOrOaccum: "); print(taccOrOaccum);
        print("\ntaccOsOaccum: "); print(taccOsOaccum);
    }

    // sOaccum is larger than sQ, so we need to syncthreads here
    // TODO: allocate enough smem for sOaccum
    if constexpr (Split) { __syncthreads(); }

    // acc_o是寄存器上的, 通過如下方式, cp到shared memory上
    cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

    const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
        + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
                                         + m_block * kBlockM) * params.d_rounded;
    const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
            ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q : bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb)
        ) + m_block * kBlockM;

    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) \
                                                + (Split ? row_offset_oaccum : row_offset_o)),
                                 Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                 make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? \
                                        params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                   Shape<Int<kBlockM>>{}, Stride<_1>{});
    // if (tidx == 0) { printf("row_offset_o = %d, bidh = %d, gOaccum = %p\n", row_offset_o, bidh, gOaccum.data()); }

    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

    if (false && threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        print("\ngOaccum: "); print(gOaccum);
        print("\ngLSEaccum: "); print(gLSEaccum);
        print("\ntOsOaccum: "); print(tOsOaccum);
        print("\ntOgOaccum: "); print(tOgOaccum);
    }

    __syncthreads();

    // 创建寄存器变量, 和global memory的变量shape一样.
    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    // shared memory上的数据, cp到register上
    cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (get<1>(taccOcO_row(0)) == 0) {
        //#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    // 构造只有形状没有类型的tensor，用于一些特定变换
    Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    // 把register里的数据cp到glbal memory上
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
    if (false && threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        print("\n=====");
        print("\ntOrOaccum: "); print(tOrOaccum);
        print("\ncaccO: "); print(caccO);
        print("\ntaccOcO: "); print(taccOcO);
        print("\ntaccOcO_row: "); print(taccOcO_row);
        print("\ncO: "); print(cO);
        print("\ntOcO: "); print(tOcO);
        print("\ntOpO: "); print(tOpO);
        //print("\ngOaccum: "); print(gOaccum); // print_tensor(gOaccum);
        //print("\nsOaccum: "); print(sOaccum); // print_tensor(sOaccum);
    }

#endif
    if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        print("\n\n");
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#if 0
template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    // We want the fwd and bwd to generate the same dropout pattern (RNG), without restricting
    // them to have the same number of threads or have to traverse the attention matrix
    // in the same order.
    // In the Philox RNG, we use the offset to store the batch, head, and the lane id
    // (within a warp). We use the subsequence to store the location of the 16 x 32 blocks within
    // the attention matrix. This way, as long as we have the batch, head, and the location of
    // the 16 x 32 block within the attention matrix, we can generate the exact same dropout pattern.

    flash::compute_attn_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, \
        bool Is_softcap, bool Split, bool Append_KV, typename Params>
inline __device__ void compute_attn_splitkv(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
    // The block index for the head.
    const int bidh = Split ? blockIdx.z - bidb * params.h : blockIdx.z;
    const int n_split_idx = Split ? blockIdx.y : 0;
    const int num_n_splits = Split ? gridDim.y : 1;
    //gridDim = {x = 1(block数), y = 5(batch), z = 12(head数)}

    flash::compute_attn_1rowblock_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, \
    Is_softcap, Split, Append_KV>(params, bidb, bidh, m_block, n_split_idx, num_n_splits);
}

////////////////////////////////////////////////////////////////////////////////////////////////////


} // namespace flash
