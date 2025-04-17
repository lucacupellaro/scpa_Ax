import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

# Load the CSV file
df = pd.read_csv('test.csv')  # Replace 'test.csv' with your actual filename

# Filter data: Mode starts with 'cuda'
df_cuda = df[df['Mode'].str.startswith('cuda', na=False)].copy()
df_cuda_csr = df[df['Mode'].str.startswith('cuda', na=False) & df['Matrix Format'].str.startswith('csr', na=False)].copy()
df_cuda_hll = df[df['Mode'].str.startswith('cuda', na=False) & df['Matrix Format'].str.startswith('hll', na=False)].copy()

print(df_cuda_csr)
# Check if there's any data after filtering
if df_cuda.empty and df_cuda_csr.empty and df_cuda_hll.empty:
    print("No data found for modes starting with 'cuda'.")
else:
    # Define nz ranges
    nz_ranges = [
        (0, 100000),
        (100000, 1000000),
        (1000000, float('inf'))  # Use float('inf') for "greater than 10000"
    ]

    # Dictionary to store maximum Measure Value and corresponding Threads for block
    max_values_info = {}
    max_values_info_csr = {}
    max_values_info_hll = {}

    # Create plots for each nz range
    for i, (nz_min, nz_max) in enumerate(nz_ranges):
        df_nz = df_cuda[(df_cuda['nz'] >= nz_min) & (df_cuda['nz'] < nz_max)].copy()
        df_nz_csr_range = df_cuda_csr[(df_cuda_csr['nz'] >= nz_min) & (df_cuda_csr['nz'] < nz_max)].copy()
        df_nz_hll_range = df_cuda_hll[(df_cuda_hll['nz'] >= nz_min) & (df_cuda_hll['nz'] < nz_max)].copy()

        if df_nz.empty and df_nz_csr_range.empty and df_nz_hll_range.empty:
            print(f"No data found for nz range: {nz_min} to {nz_max}")
            continue  # Skip to the next range if no data

        # Plot for all CUDA modes
        if not df_nz.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.lineplot(
                x='Threads for block',
                y='Measure Value',
                hue='Matrix Name',
                style='Mode',
                data=df_nz,
                marker='o',
                ax=ax,
            )
            ax.set_title(f'Measure Value vs. Threads for Block (CUDA Modes) - nz: {nz_min} to {nz_max}')
            ax.set_xlabel('Threads for Block')
            ax.set_ylabel('Measure Value')
            ax.set_xticks([2, 4, 8, 16, 32, 64, 128])  # Set specific x-ticks
            ax.set_xlim(1, 128)  # Keep x-axis limits for context
            ax.grid(True, which="both", ls="--", alpha=0.6)
            ax.set_facecolor('white')
            fig.tight_layout()
            plt.show()

            # Calculate and store the maximum Measure Value and corresponding Threads for block (all CUDA)
            for matrix_name in df_nz['Matrix Name'].unique():
                for mode in df_nz['Mode'].unique():
                    function_label = f"{matrix_name} - {mode} "
                    max_row = df_nz[(df_nz['Matrix Name'] == matrix_name) & (df_nz['Mode'] == mode)].nlargest(1, 'Measure Value')
                    if not max_row.empty:
                        max_value = max_row['Measure Value'].values[0]
                        max_threads = max_row['Threads for block'].values[0]  # Corrected column name
                        max_values_info[function_label] = {'max_value': max_value, 'max_threads': max_threads}
                    else:
                        max_values_info[function_label] = {'max_value': None, 'max_threads': None}

        # Plot for CUDA modes with CSR format
        if not df_nz_csr_range.empty:
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            sns.lineplot(
                x='Threads for block',
                y='Measure Value',
                hue='Matrix Name',
                style='Mode',
                data=df_nz_csr_range,
                marker='o',
                ax=ax1,
            )
            ax1.set_title(f'Measure Value vs. Threads for Block (CUDA CSR Modes) - nz: {nz_min} to {nz_max}')
            ax1.set_xlabel('Threads for Block')
            ax1.set_ylabel('Measure Value')
            ax1.set_xticks([2, 4, 8, 16, 32, 64, 128])  # Set specific x-ticks
            ax1.set_xlim(1, 128)  # Keep x-axis limits for context
            ax1.grid(True, which="both", ls="--", alpha=0.6)
            ax1.set_facecolor('white')
            fig1.tight_layout()
            plt.show()

            # Calculate and store the maximum Measure Value and corresponding Threads for block (CUDA CSR)
            for matrix_name in df_nz_csr_range['Matrix Name'].unique():
                for mode in df_nz_csr_range['Mode'].unique():
                    function_label = f"{matrix_name} - {mode} (CSR)"
                    max_row_csr = df_nz_csr_range[(df_nz_csr_range['Matrix Name'] == matrix_name) & (df_nz_csr_range['Mode'] == mode)].nlargest(1, 'Measure Value')
                    if not max_row_csr.empty:
                        max_value_csr = max_row_csr['Measure Value'].values[0]
                        max_threads_csr = max_row_csr['Threads for block'].values[0]  # Corrected column name
                        max_values_info_csr[function_label] = {'max_value': max_value_csr, 'max_threads': max_threads_csr}
                    else:
                        max_values_info_csr[function_label] = {'max_value': None, 'max_threads': None}

        # Plot for CUDA modes with HLL format
        if not df_nz_hll_range.empty:
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            sns.lineplot(
                x='Threads for block',
                y='Measure Value',
                hue='Matrix Name',
                style='Mode',
                data=df_nz_hll_range,
                marker='o',
                ax=ax2,
            )
            ax2.set_title(f'Measure Value vs. Threads for Block (CUDA HLL Modes) - nz: {nz_min} to {nz_max}')
            ax2.set_xlabel('Threads for Block')
            ax2.set_ylabel('Measure Value')
            ax2.set_xticks([2, 4, 8, 16, 32, 64, 128])  # Set specific x-ticks
            ax2.set_xlim(1, 128)  # Keep x-axis limits for context
            ax2.grid(True, which="both", ls="--", alpha=0.6)
            ax2.set_facecolor('white')
            fig2.tight_layout()
            plt.show()

            # Calculate and store the maximum Measure Value and corresponding Threads for block (CUDA HLL)
            for matrix_name in df_nz_hll_range['Matrix Name'].unique():
                for mode in df_nz_hll_range['Mode'].unique():
                    function_label = f"{matrix_name} - {mode} (HLL)"
                    max_row_hll = df_nz_hll_range[(df_nz_hll_range['Matrix Name'] == matrix_name) & (df_nz_hll_range['Mode'] == mode)].nlargest(1, 'Measure Value')
                    if not max_row_hll.empty:
                        max_value_hll = max_row_hll['Measure Value'].values[0]
                        max_threads_hll = max_row_hll['Threads for block'].values[0]  # Corrected column name
                        max_values_info_hll[function_label] = {'max_value': max_value_hll, 'max_threads': max_threads_hll}
                    else:
                        max_values_info_hll[function_label] = {'max_value': None, 'max_threads': None}

    # Print the maximum Measure Values and Corresponding Threads for Block
    print("\nMaximum Measure Values and Corresponding Threads for Block (All CUDA Modes):")
    for function, info in max_values_info.items():
        print(f"{function}: Max Value = {info['max_value']:.4f}, Threads = {info['max_threads']}")

    print("\nMaximum Measure Values and Corresponding Threads for Block (CUDA CSR Modes):")
    for function, info in max_values_info_csr.items():
        print(f"{function}: Max Value = {info['max_value']:.4f}, Threads = {info['max_threads']}")

    print("\nMaximum Measure Values and Corresponding Threads for Block (CUDA HLL Modes):")
    for function, info in max_values_info_hll.items():
        print(f"{function}: Max Value = {info['max_value']:.4f}, Threads = {info['max_threads']}")