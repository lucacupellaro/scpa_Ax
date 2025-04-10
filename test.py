def generate_zero_matrix_txt(rows, cols):
    """
    Generates a text file representing a sparse matrix where all values are 0.
    Since all values are zero, the file will only contain the dimensions
    and the number of non-zero entries (which is 0).

    Args:
        rows (int): The number of rows in the matrix.
        cols (int): The number of columns in the matrix.
    """
    num_non_zero = 1296 # Because all values are 0

    with open("zero_matrix.txt", "w") as f:
        f.write(f"{rows} {cols} {num_non_zero}\n")
        for i in range(1,rows+1):
            for j in range(1,cols+1):
                    f.write(f"{i} {j} {0.5}\n")



if __name__ == "__main__":
    num_rows = 36
    num_cols = 36
    generate_zero_matrix_txt(num_rows, num_cols)
    print(f"A {num_rows}x{num_cols} zero matrix representation has been written to zero_matrix.txt")
