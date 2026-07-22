import pytest
import torch
import triton
import triton.language as tl
import numpy as np
import test_common
from triton.tools.get_ascend_devices import is_compile_on_910_95

# Chained dot_scaled: first matmul with zero init, second accumulates via acc=.
# Mirrors normalize-matmul.mlir @test_mmadmx_chain_no_elemwise_decompose:
#   mmadmxL1(init=true,  outs=empty) -> mmadmxL1(init=false, outs=first)
# No transpose. Result should equal 2 * dot_scaled(lhs, rhs, scales).


def ub_overflow_check(M, N, K):
    bytes_of_dtype = 4
    M32 = ((M + 31) // 32) * 32
    N32 = ((N + 31) // 32) * 32
    cond1 = M32 * K * bytes_of_dtype <= 60 * 1024
    cond2 = K * N32 * bytes_of_dtype <= 60 * 1024
    cond3 = M32 * N32 * bytes_of_dtype <= 256 * 1024
    return cond1 and cond2 and cond3


def torch_dot_scaled(a, b, sa, sb):
    sa = torch.exp2(sa - 127).repeat_interleave(32, dim=1)
    sb = torch.exp2(sb - 127).repeat_interleave(32, dim=1).T
    return torch.matmul(a * sa, b * sb)


@triton.jit
def dot_scale_chain_kernel(
    lhs_ptr,
    lhs_scale_ptr,
    rhs_ptr,
    rhs_scale_ptr,
    out_ptr,
    lhs_format: tl.constexpr,
    rhs_format: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    Midx = tl.arange(0, M)
    Nidx = tl.arange(0, N)
    Kidx = tl.arange(0, K)
    K1idx = tl.arange(0, K // 32)

    lhs = tl.load(lhs_ptr + Midx[:, None] * K + Kidx[None, :])
    lhs_scale = tl.load(lhs_scale_ptr + Midx[:, None] * (K // 32) + K1idx[None, :])
    rhs = tl.load(rhs_ptr + Kidx[:, None] * N + Nidx[None, :])
    rhs_scale = tl.load(rhs_scale_ptr + Nidx[:, None] * (K // 32) + K1idx[None, :])

    # First mmad: init=true equivalent (fresh accumulation).
    first = tl.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format)
    # Second mmad: init=false equivalent (accumulate into L0C).
    result = tl.dot_scaled(
        lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc=first)
    tl.store(out_ptr + Midx[:, None] * N + Nidx[None, :], result)


testlist = [
    # K = 64 — very small / edge
    (1, 1, 64),
    (1, 7, 64),
    (4, 16, 64),
    (7, 9, 64),
    # K = 64 — small irregular (< 64)
    (11, 13, 64),
    (12, 64, 64),
    (15, 128, 64),
    (16, 16, 64),
    (17, 23, 64),
    (27, 15, 64),
    (33, 17, 64),
    (33, 33, 64),
    # K = 64 — near 32 / 64 boundaries
    (31, 29, 64),
    (32, 32, 64),
    (32, 64, 64),
    (35, 37, 64),
    (37, 21, 64),
    (41, 53, 64),
    (45, 23, 64),
    (47, 113, 64),
    (48, 64, 64),
    (59, 61, 64),
    (63, 59, 64),
    (64, 32, 64),
    (64, 48, 64),
    (64, 64, 64),
    (67, 71, 64),
    # K = 64 — medium irregular
    (73, 45, 64),
    (85, 57, 64),
    (87, 199, 64),
    (94, 64, 64),
    (95, 97, 64),
    (97, 93, 64),
    (98, 64, 64),
    (101, 103, 64),
    (113, 47, 64),
    # K = 64 — around 128
    (127, 121, 64),
    (128, 64, 64),
    (129, 129, 64),
    (135, 119, 64),
    # K = 64 — larger irregular
    (157, 141, 64),
    (189, 175, 64),
    (199, 87, 64),
    (211, 97, 64),
    (223, 111, 64),
    (223, 223, 64),
    # K = 128 — small irregular
    (1, 1, 128),
    (1, 96, 128),
    (3, 17, 128),
    (5, 23, 128),
    (7, 7, 128),
    (7, 9, 128),
    (9, 91, 128),
    (11, 13, 128),
    (13, 83, 128),
    (15, 95, 128),
    (16, 16, 128),
    (17, 79, 128),
    (19, 23, 128),
    (21, 37, 128),
    (23, 19, 128),
    (27, 15, 128),
    (29, 31, 128),
    (31, 29, 128),
    (33, 33, 128),
    # K = 128 — medium irregular
    (35, 37, 128),
    (37, 21, 128),
    (41, 43, 128),
    (43, 41, 128),
    (45, 23, 128),
    (47, 53, 128),
    (51, 59, 128),
    (53, 47, 128),
    (55, 61, 128),
    (59, 51, 128),
    (61, 55, 128),
    (63, 59, 128),
    (64, 17, 128),
    (64, 32, 128),
    (64, 64, 128),
    (67, 71, 128),
    (71, 67, 128),
    (73, 45, 128),
    (77, 83, 128),
    (79, 17, 128),
    (81, 89, 128),
    (83, 77, 128),
    (85, 57, 128),
    (87, 91, 128),
    (89, 87, 128),
    (91, 93, 128),
    (93, 91, 128),
    # K = 128 — near UB limit / asymmetric
    (95, 15, 128),
    (95, 96, 128),
    (96, 1, 128),
    (96, 95, 128),
    (96, 96, 128),
    # K = 192 — small irregular
    (1, 1, 192),
    (1, 64, 192),
    (3, 17, 192),
    (5, 23, 192),
    (7, 9, 192),
    (9, 55, 192),
    (11, 13, 192),
    (13, 47, 192),
    (15, 63, 192),
    (17, 64, 192),
    (19, 23, 192),
    (21, 37, 192),
    (23, 19, 192),
    (27, 31, 192),
    (29, 31, 192),
    (31, 27, 192),
    # K = 192 — medium irregular
    (32, 32, 192),
    (33, 17, 192),
    (35, 37, 192),
    (37, 41, 192),
    (41, 37, 192),
    (43, 47, 192),
    (45, 23, 192),
    (47, 43, 192),
    (49, 53, 192),
    (51, 59, 192),
    (53, 49, 192),
    (55, 9, 192),
    (57, 61, 192),
    (59, 51, 192),
    (61, 57, 192),
    # K = 192 — near UB limit / asymmetric
    (62, 64, 192),
    (63, 63, 192),
    (64, 1, 192),
    (64, 17, 192),
    (64, 62, 192),
    (64, 64, 192),
    # K = 256 — small irregular
    (1, 1, 256),
    (1, 32, 256),
    (3, 17, 256),
    (5, 23, 256),
    (7, 7, 256),
    (7, 25, 256),
    (9, 31, 256),
    (11, 13, 256),
    (13, 27, 256),
    (15, 29, 256),
    (16, 16, 256),
    (17, 23, 256),
    (19, 31, 256),
    (21, 11, 256),
    (23, 17, 256),
    # K = 256 — medium irregular / near UB limit
    (25, 7, 256),
    (27, 31, 256),
    (29, 15, 256),
    (31, 9, 256),
    (31, 27, 256),
    (31, 31, 256),
    (32, 1, 256),
    (32, 17, 256),
    (32, 32, 256),
]


def generate_tensor_fp8(shape,
                        dtype,
                        seed=None,
                        data_range=None,
                        special_ratio=0.05,  # 插入 inf/-inf/nan的比例（仅当dtype为float类时生效）
                        precision_ratio=0.35,  # 精度小浮点数的比例（仅当dtype为float类时生效）
                        extreme_ratio=0.1,  # 极值区域占比（其中20%为 min~min+10, max-10~max，80%为min和max本身）（dtype为bool时不生效）
                        zero_ratio=0,  # 额外加入 0 的比例 （dtype为bool时不生效）
                        bool_true_ratio=0.5,  # True 的占比 （仅当dtype为bool时生效）
                        ):
    middle_range_dict = {
        'fp8e4m3': [-50, 50],
        'fp8e5m2': [-500, 500],
        'fp8e5b16': [-500, 500],
        'fp4': [-6, 6],
    }

    float_precision_dict = {
        'float16': [-1, 1],
        'float32': [-1, 1],
        'bfloat16': [-1, 1],
        'fp8e4m3': [-1, 1],
        'fp8e5m2': [-1, 1],
        'fp8e5b16': [-1, 1],
        'fp4': [-2, 2],
    }

    extreme_range_dict = {
        'fp8e4m3': [-448, 448],
        'fp8e5m2': [-57344, 57344],
        'fp8e5b16': [-57344, 57344],
        'fp4': [-6, 6],
    }

    ddtype_dict = {
        'fp8e4m3': torch.float8_e4m3fn,
        'fp8e5m2': torch.float8_e5m2,
        'fp8e5b16': None,
        'fp4': None,
    }

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if dtype not in ddtype_dict:
        raise ValueError(f"Unsupported dtype: {dtype}")

    min_val, max_val = extreme_range_dict[dtype]
    total = np.prod(shape)

    if dtype == 'fp4':
        left_range = (min_val, min_val + 1)
        right_range = (max_val - 1, max_val)
    else:
        left_range = (min_val, min_val + 10)
        right_range = (max_val - 10, max_val)

    if data_range == None:
        mid_range = middle_range_dict[dtype]
    else:
        mid_range = data_range

    # float类处理
    if dtype in ('float16', 'float32', 'bfloat16', 'fp8e4m3', 'fp8e5m2'):

        # 指定分布概率和超过100%异常处理
        if (extreme_ratio + precision_ratio + zero_ratio + special_ratio > 1):
            raise ValueError(f"Total ratio over 100%, unable to allocate sampling:{extreme_ratio + precision_ratio}")

        # 读字典取得精度小浮点数数据范围
        precision_range = float_precision_dict[dtype]

        # 生成原始一维全0数据，float32防溢出
        data = np.zeros(total, dtype=np.float32)

        # 计算各区间元素个数（其中极值区域为累计计数：按极小值附近，极小值本身，极大值附近，极大值本身顺序）
        left_count = int(total * extreme_ratio / 10)
        left_accumulate_count = int(total * extreme_ratio / 2)
        left_right_count = int(total * extreme_ratio * 3 / 5)
        left_right_accmulate_count = int(total * extreme_ratio)
        precision_count = int(total * precision_ratio)
        zero_count = int(total * zero_ratio)
        special_count = int(total * special_ratio)

        # 生成打乱的索引映射数组
        indices = np.random.choice(total, size=total, replace=False)

        # 按区间元素个数划分区间索引
        left_idx = indices[:left_count]
        left_min_idx = indices[left_count:left_accumulate_count]
        right_idx = indices[left_accumulate_count:left_right_count]
        right_max_idx = indices[left_right_count:left_right_accmulate_count]
        precision_idx = indices[left_right_accmulate_count:left_right_accmulate_count + precision_count]
        zero_idx = indices[left_right_accmulate_count + precision_count:
                           left_right_accmulate_count + precision_count + zero_count]
        special_idx = indices[left_right_accmulate_count + precision_count + zero_count:
                              left_right_accmulate_count + precision_count + zero_count + special_count]
        mid_idx = indices[left_right_accmulate_count + precision_count + zero_count + special_count:]

        # 按区间索引生成数据
        data[left_idx] = np.random.uniform(left_range[0], left_range[1], size=len(left_idx))
        data[left_min_idx] = min_val
        data[right_idx] = np.random.uniform(right_range[0], right_range[1], size=len(right_idx))
        data[right_max_idx] = max_val
        data[precision_idx] = np.random.uniform(precision_range[0], precision_range[1], size=len(precision_idx))
        data[zero_idx] = 0
        for idx in special_idx:
            r = np.random.rand()
            if r < 1 / 3:
                data[idx] = float('inf')
            elif r < 2 / 3:
                data[idx] = float('-inf')
            else:
                data[idx] = float('nan')
        data[mid_idx] = np.random.uniform(mid_range[0], mid_range[1], size=len(mid_idx))

        tensor = torch.from_numpy(data.reshape(shape)).to(ddtype_dict[dtype])

    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return tensor


@pytest.mark.skipif(not is_compile_on_910_95, reason="A5脚本，执行跳过")
@pytest.mark.parametrize("dtype", ["fp8e4m3", "fp8e5m2"])
@pytest.mark.parametrize("M, N, K", testlist)
def test_2D_scaled_dot_chain(M, N, K, dtype):
    if not ub_overflow_check(M, N, K):
        pytest.skip("UB overflow")

    lhs_format = dtype[3:]
    rhs_format = lhs_format

    lhs = generate_tensor_fp8(shape=(M, K), dtype=dtype).npu()
    rhs = generate_tensor_fp8(shape=(K, N), dtype=dtype).npu()
    lhs_scale = torch.randint(low=110, high=128, size=(M, K // 32), dtype=torch.int8).npu()
    rhs_scale = torch.randint(low=110, high=128, size=(N, K // 32), dtype=torch.int8).npu()

    lhs_f32 = lhs.cpu().to(torch.float32)
    rhs_f32 = rhs.cpu().to(torch.float32)
    lhs_scale_f32 = lhs_scale.cpu().to(torch.float32)
    rhs_scale_f32 = rhs_scale.cpu().to(torch.float32)

    triton_res = torch.zeros((M, N), dtype=torch.float32).npu()
    dot_scale_chain_kernel[(1,)](
        lhs, lhs_scale, rhs, rhs_scale, triton_res,
        lhs_format, rhs_format, M, N, K)

    single = torch_dot_scaled(lhs_f32, rhs_f32, lhs_scale_f32, rhs_scale_f32)
    torch_ref = (single + single).to(torch.float32)

    if dtype == "fp8e4m3":
        torch.testing.assert_close(
            triton_res.cpu(), torch_ref.cpu(),
            rtol=125e-03, atol=125e-03, equal_nan=True)
    elif dtype == "fp8e5m2":
        torch.testing.assert_close(
            triton_res.cpu(), torch_ref.cpu(),
            rtol=75e-02, atol=75e-02, equal_nan=True)


