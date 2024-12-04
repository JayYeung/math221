import torch

def converter(matrix, blocksize=16):
    """
    Converts a dense matrix into Block Compressed Sparse Row (BCSR) format.

    Args:
        matrix (torch.Tensor): The dense matrix to convert.
        blocksize (int): The size of the square blocks.

    Returns:
        tuple: (crow_indices, col_indices, values)
    """
    # Ensure the matrix dimensions are divisible by the block size
    assert matrix.size(0) % blocksize == 0 and matrix.size(1) % blocksize == 0, \
        "Matrix dimensions must be divisible by block size."
    
    num_blocks_row = matrix.size(0) // blocksize
    num_blocks_col = matrix.size(1) // blocksize

    crow_indices = [0]
    col_indices = []
    values = []

    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            block = matrix[
                i * blocksize : (i + 1) * blocksize,
                j * blocksize : (j + 1) * blocksize
            ]
            # Check if the block is non-zero
            if not torch.all(block == 0):
                col_indices.append(j)
                values.append(block.clone())  
        crow_indices.append(len(col_indices))
    
    return crow_indices, col_indices, values


if __name__ == "__main__":
    matrix = torch.tensor([[1, 2, 0, 0], 
                        [0, 3, 0, 0], 
                        [0, 0, 4, 0], 
                        [0, 0, 5, 6]])
    crow_indices, col_indices, values = converter(matrix, blocksize=2)
    print("crow_indices:", crow_indices)
    print("col_indices:", col_indices)
    print("values:", values)

    # crow_indices: [0, 1, 2]
    # col_indices: [0, 1]
    # values: [tensor([[1, 2],
    #                  [0, 3]]), 
    #          tensor([[4, 0],
    #                  [5, 6]])]