import triton
import triton.language as tl


@triton.jit
def flash_attention(
    key_mat,
    query_mat,
    values_mat,
    out_mat,
    key_strides_x: int,
    key_strides_y: int,
    query_strides_x: int,
    query_strides_y: int,
    values_strides_x: int,
    values_strides_y: int,
    out_strides_x: int,
    out_strides_y: int,
    n_dim: int,
    d_dim: int,
    **META
):
    BLOCK_M = META["BLOCK_M"]
    BLOCK_N = META["BLOCK_N"]
    BLOCK_K = META['BLOCK_K']

    pid_m = tl.program_id(0) # Zeile
    pid_n = tl.program_id(1) # Spalte

    # indice range
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # base pointer initialization
    base_key_mat = key_mat + (rm[:, None] * key_strides_y + rk[None, :] * key_strides_x)
    base_query_mat = query_mat + (rk[:, None] * query_strides_y + rn[None, :] * query_strides_x)
    base_value_mat = values_mat + (rm[:, None] * values_strides_y + rn[None, :] * values_strides_x)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    global_m = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    global_l = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k in range(n_dim, 0, -BLOCK_K):
        key_mat_loaded = tl.load(base_key_mat) # load key matrix block
        query_mat_loaded = tl.load(base_query_mat) # load query matrix block
        values_mat_loaded = tl.load(base_value_mat) # load values matrix block

        s = tl.dot(query_mat_loaded, tl.trans(key_mat_loaded))
        s_scaled = s / tl.sqrt(d_dim)

        m = tl.max(s_scaled, axis=1)
        new_global_m = tl.maximum(global_m, m)
        alpha = tl.exp(global_m - new_global_m)

        P = tl.exp(s_scaled - global_m[:, None])

        global_l = alpha * global_l + tl.sum(P, axis=1)
        acc_scaled = acc * alpha[:, None]
        acc = acc_scaled + tl.dot(P, values_mat_loaded)

        global_m = new_global_m # set new global maximum

        base_key_mat += BLOCK_K * key_strides_x # shift key matrix block
        base_value_mat += BLOCK_K * values_strides_x # shift value matrix block

    out = out_mat + (rm[:, None] * out_strides_y + rn[None, :] * out_strides_x)
    mask = (rm[:, None] < n_dim) & (rn[None, :] < d_dim) # mask for
    tl.store(out, acc / global_l[:, None], mask=mask) # save block into global memory