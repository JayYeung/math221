import torch

def to_bsr(matrix, block_size):
    num_blocks_row = matrix.size(0) // block_size
    num_blocks_col = matrix.size(1) // block_size

    crow_indices = [0]
    col_indices = []
    values = []

    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            block = matrix[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size
            ]

            if not torch.all(block <= 1e-16):
                col_indices.append(j)
                values.append(block.clone())

        crow_indices.append(len(col_indices))

    return crow_indices, col_indices, values
