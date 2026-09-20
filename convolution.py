import triton
import triton.language as tl


@triton.jit
def convolution(
        mat,
        kernel,
        out,
        mat_y,
        mat_x,
        kernel_dim,
        stride_ax,
        stride_ay,
        stride_out_y,
        stride_out_x,
        **META):
    # Meta Parameter
    BLOCK_M, GROUP_M = META['BLOCK_M'], META['GROUP_M']
    BLOCK_N = META['BLOCK_N']

    pid_m = tl.program_id(0) # Zeile
    pid_n = tl.program_id(1) # Spalte

    # Indice range
    ry = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) # array with starting memory address for row
    rx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) # array with starting memory address for col

    mat = mat + (ry[:, None] * stride_ay + rx[None, :] * stride_ax) # memory addresses

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for ky in range(kernel_dim):
        for kx in range(kernel_dim):
            kernel_ptr = kernel + (ky * kernel_dim + kx) # kernel[ky, kx]
            weight = tl.load(kernel_ptr) # loading one weight from the kernel

            current_mat_ptr = mat + (ky * stride_ay + kx * stride_ax) # loading current mat section (starting point)

            mask = (ry[:, None] + ky < mat_y) & (rx[None, :] + kx < mat_x) # masking for overlapping pixel
            image_block = tl.load(current_mat_ptr, mask=mask, other=0.0) # loading of the current block

            acc += tl.mul(image_block, weight)

    out_ptr = out + (ry[:, None] * stride_out_y + rx[None, :] * stride_out_x)
    out_mask = (ry[:, None] < mat_y) & (rx[None, :] < mat_x)
    tl.store(out_ptr, acc, mask=out_mask)
