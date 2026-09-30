import triton
import triton.language as tl


class ConvHardswish:
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 16}, num_warps=2, num_stages=2),
            triton.Config({'BLOCK_SIZE': 32}, num_warps=4, num_stages=2),
            triton.Config({'BLOCK_SIZE': 64}, num_warps=4, num_stages=4),
        ],
        key=['mat_y', 'mat_x'],
    )

    @triton.jit
    @staticmethod
    def kernel(
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
                current_kernel_ptr = kernel_ptr + (ky * kernel_dim + kx)  # kernel[ky, kx]
                weight = tl.load(current_kernel_ptr)  # loading one weight from the kernel

                current_mat_ptr = mat + (ky * stride_ay + kx * stride_ax)  # loading current mat section (starting point)

                mask = (ry[:, None] + ky < mat_y) & (rx[None, :] + kx < mat_x)  # masking for overlapping pixel
                image_block = tl.load(current_mat_ptr, mask=mask, other=0.0)  # loading of the current block

                # convolution
                acc += tl.mul(image_block, weight)

        # hardswish activation
        acc = tl.where(
            acc <= -3.0,
            0,
            tl.where(
                acc >= 3.0,
                acc,
                tl.div_rn(tl.mul(acc, (acc + 3.0)), 6.0)
            )
        )

        out_ptr = out_ptr + (ry[:, None] * stride_out_y + rx[None, :] * stride_out_x)
        out_mask = (ry[:, None] < mat_y) & (rx[None, :] < mat_x)
        tl.store(out_ptr, acc, mask=out_mask)

