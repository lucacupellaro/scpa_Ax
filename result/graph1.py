import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Imposta lo stile ggplot per un aspetto semplice e pulito
plt.style.use('ggplot')

# Carica il file CSV
df = pd.read_csv('test.csv')

# Visualizzazioni suggerite:

# 1. Line plot del Measure Value rispetto al Measure Index, separato per Mode e Matrix Format
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Measure Index', y='Measure Value', hue='Mode', style='Matrix Format', marker='o')
plt.title('Measure Value vs Measure Index per Mode e Matrix Format')
plt.xlabel('Measure Index')
plt.ylabel('Measure Value')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# 2. Bar plot del Measure Value medio per Mode, separato per Matrix Format
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='Matrix Format', y='Measure Value', hue='Mode')
plt.title('Valore Medio della Misura per Mode e Matrix Format')
plt.xlabel('Matrix Format')
plt.ylabel('Valore Medio della Misura')
plt.legend(title='Mode')
plt.tight_layout()
plt.show()

# 3. Box plot del Measure Value per Mode, separato per Matrix Format
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='Matrix Format', y='Measure Value', hue='Mode')
plt.title('Distribuzione del Valore della Misura per Mode e Matrix Format')
plt.xlabel('Matrix Format')
plt.ylabel('Measure Value')
plt.legend(title='Mode')
plt.tight_layout()
plt.show()

# 4. Scatter plot del Measure Value rispetto al Number of Threads, separato per Mode e Matrix Format
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Number of Threads', y='Measure Value', hue='Mode', style='Matrix Format', s=100)
plt.title('Measure Value vs Number of Threads per Mode e Matrix Format')
plt.xlabel('Number of Threads')
plt.ylabel('Measure Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Facet Grid per visualizzare Measure Value vs Measure Index per combinazioni di Mode e Matrix Format
g = sns.FacetGrid(df, col='Mode', row='Matrix Format', height=4, aspect=1.5)
g.map(sns.lineplot, 'Measure Index', 'Measure Value', marker='o')
g.add_legend()
g.set_axis_labels('Measure Index', 'Measure Value')
g.fig.suptitle('Measure Value vs Measure Index per Mode e Matrix Format', y=1.02)
plt.tight_layout()
plt.show()
