import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Imposta lo stile ggplot per un aspetto semplice e pulito
plt.style.use('ggplot')

# Carica il file CSV
df = pd.read_csv('test.csv')

# Filtra il DataFrame per includere solo le righe con Mode 'serial', 'openMp'
modes_to_plot = ['serial', 'openMp']
df_filtered = df[df['Mode'].isin(modes_to_plot)].copy()

# Calcola la media dei 'Measure Value' per ogni 'Matrix Name', 'Mode' e 'Matrix Format'
average_performance_combined = df_filtered.groupby(['Matrix Name', 'Mode', 'Matrix Format'])['Measure Value'].mean().reset_index()

# Crea una nuova colonna per la legenda combinata
def create_combined_legend_label(row):
    return f"{row['Mode']} ({row['Matrix Format']})"

average_performance_combined['Legend Label'] = average_performance_combined.apply(create_combined_legend_label, axis=1)

# Crea il grouped bar plot combinato
plt.figure(figsize=(14, 8))
sns.barplot(x='Matrix Name', y='Measure Value', hue='Legend Label', data=average_performance_combined)
plt.title('Average Performance by Matrix and Mode (OpenMP CSR vs OpenMP HLL vs Serial)')
plt.xlabel('Matrix Name')
plt.ylabel('Average Measure Value')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Mode (Format)')
plt.tight_layout()
plt.show()

# Funzioni originali per il grafico semplice (mantenute per compatibilità)
def calculate_average_performance(df):
    """Calculates the average 'Measure Value' for each 'Matrix Name' and 'Mode'."""
    average_performance = df.groupby(['Matrix Name', 'Mode'])['Measure Value'].mean().reset_index()
    return average_performance

average_performance_df = calculate_average_performance(df_filtered)

def prepare_plotting_data(average_performance):
    """Prepares data for plotting by pivoting the DataFrame."""
    plotting_data = average_performance.pivot(index='Matrix Name', columns='Mode', values='Measure Value').reset_index()
    plotting_data = plotting_data.melt(id_vars='Matrix Name', var_name='Mode', value_name='Average Measure Value')
    return plotting_data

plotting_df = prepare_plotting_data(average_performance_df)

def create_grouped_bar_plot(plotting_df):
    """Creates a grouped bar plot to visualize average performance."""
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Matrix Name', y='Average Measure Value', hue='Mode', data=plotting_df)
    plt.title('Average Performance by Matrix and Mode')
    plt.xlabel('Matrix Name')
    plt.ylabel('Average Measure Value')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

create_grouped_bar_plot(plotting_df)


# Filtra per Matrix Format (csr o hll)
df_filtered = df[df['Matrix Format'].isin(['csr', 'hll'])].copy()

# Filtra per Mode che inizia con 'cuda'
df_filtered = df_filtered[df_filtered['Mode'].str.startswith('cuda', na=False)].copy()

# Verifica se ci sono dati dopo il filtraggio
if df_filtered.empty:
    print("Nessun dato corrispondente ai criteri di filtro.")
else:
    # Calcola la media di 'Measure Value' per ogni 'Matrix Name' e 'Mode'
    average_performance = df_filtered.groupby(['Matrix Name', 'Mode', 'Matrix Format'])['Measure Value'].mean().reset_index()

    # Crea un grafico a barre per ogni Matrix Format
    for matrix_format in average_performance['Matrix Format'].unique():
        df_format = average_performance[average_performance['Matrix Format'] == matrix_format]

        plt.figure(figsize=(12, 6))  # Adjust figure size as needed
        sns.barplot(x='Matrix Name', y='Measure Value', hue='Mode', data=df_format)
        plt.title(f'Average Measure Value for {matrix_format} (CUDA Modes)')
        plt.xlabel('Matrix Name')
        plt.ylabel('Average Measure Value')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
        plt.legend(title='Mode')
        plt.tight_layout()
        plt.show()

