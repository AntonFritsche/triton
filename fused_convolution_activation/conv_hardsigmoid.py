import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            kwargs={'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=2, num_stages=2
        ),
        triton.Config(
            kwargs={'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4, num_stages=2
        ),
        triton.Config(
            kwargs={'BLOCK_M': 256, 'BLOCK_N': 256}, num_warps=6, num_stages=4
        ),
        triton.Config(
            kwargs={'BLOCK_M': 512, 'BLOCK_N': 512}, num_warps=8, num_stages=4
        ),
        triton.Config(
            kwargs={'BLOCK_M': 1024, 'BLOCK_N': 1024}, num_warps=10, num_stages=6
        ),
    ],
    key=['mat_x', 'mat_y'],
)

@triton.jit
def conv_elu(
        mat_ptr,
        kernel_ptr,
        out_ptr,
        mat_y: int,
        mat_x: int,
        kernel_dim: int,
        stride_ax: int,
        stride_ay: int,
        stride_out_y: int,
        stride_out_x: int,
        BLOCK_SIZE: tl.constexpr,
):
    # META Parameters
    BLOCK_M = BLOCK_SIZE
    BLOCK_N = BLOCK_SIZE

    pid_m = tl.program_id(0)  # Zeile
    pid_n = tl.program_id(1)  # Spalte

    # Indice range
    ry = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # array with starting memory address for row
    rx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # array with starting memory address for col

    mat = mat_ptr + (ry[:, None] * stride_ay + rx[None, :] * stride_ax)  # memory addresses

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for ky in range(kernel_dim):
        for kx in range(kernel_dim):
            kernel_ptr = kernel_ptr + (ky * kernel_dim + kx)  # kernel[ky, kx]
            weight = tl.load(kernel_ptr)  # loading one weight from the kernel

            current_mat_ptr = mat + (ky * stride_ay + kx * stride_ax)  # loading current mat section (starting point)

            mask = (ry[:, None] + ky < mat_y) & (rx[None, :] + kx < mat_x)  # masking for overlapping pixel
            image_block = tl.load(current_mat_ptr, mask=mask, other=0.0)  # loading of the current block

            # convolution
            acc += tl.mul(image_block, weight)

    # hardsigmoid activation
    acc = tl.where(
        acc <= -3.0,
        0,
        tl.where(
            acc >= 3.0,
            1,
            tl.div_rn(acc, 6.0) + 0.5
        )
    )

    acc = tl.where(-3.0 < acc < 3.0, tl.div_rn(acc, 6.0) + 0.5, acc)
    out_ptr = out_ptr + (ry[:, None] * stride_out_y + rx[None, :] * stride_out_x)
    out_mask = (ry[:, None] < mat_y) & (rx[None, :] < mat_x)
    tl.store(out_ptr, acc, mask=out_mask)
