import torch
from torch.sparse import _triton_ops


def create_blocked_mask(self, shape, p, block_size):
    """
    Create a blocked sparse mask for a given shape and proportion p.
    """
    mask = torch.ones(shape)

    # Divide the weight matrix into blocks
    rows, cols = shape
    row_blocks = rows // block_size
    col_blocks = cols // block_size
    
    zeroed = 0

    for i in range(row_blocks):
        for j in range(col_blocks):
            if torch.rand(1).item() < p:
                mask[
                    i * block_size : (i + 1) * block_size,
                    j * block_size : (j + 1) * block_size,
                ] = 0
                zeroed += block_size ** 2
    
    # print(f'Zeroed {zeroed} out of {rows * cols} weights, proportion = {zeroed / (rows * cols)} ({float(p)})')
    
    return mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_bsr1 = (create_blocked_mask(None, (4096, 4096), 0.85, 32) * \
    torch.rand(4096, 4096)).to_sparse_bsr(32).to(device), \
    torch.rand(4096, 4096).to(device)

print(type(_triton_ops.bsr_dense_mm))

# print(_triton_ops(
#     (create_blocked_mask(None, (4096, 4096), 0.85, 32) * 
#      torch.rand(4096, 4096)).to_sparse_bsr(32).to(device),
#     torch.rand(4096, 4096).to(device))
# )


