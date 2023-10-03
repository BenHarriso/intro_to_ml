import pandas as pd
df = pd.read_csv("iris.csv")

import matplotlib.pyplot as plt
plt.plot(df['sepal_length'])
plt.title('Sepal-Length-Over-Index')
plt.xlabel('Index')
plt.ylabel('Sepal-Length-(cm)')
plt.savefig('line_plot.png')

plt.hist(df['sepal_length'], bins=10)
plt.xticks([4, 5, 6, 7, 8])
plt.title('Histogram of Sepal-Length')
plt.xlabel('Sepal-Length-(cm)')
plt.ylabel('Frequency')
plt.savefig('histogram.png')

species_count = df['class'].value_counts()
plt.pie(species_count, labels=species_count.index, autopct='%1.1f%%')

plt.title('Species Distribution')
plt.savefig('pie_chart.png')

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
colors = {'Iris-setosa': 'red', 'Iris-versicolor':'green', 'Iris-virginica':'blue'}
markers = {'Iris-setosa': 'o', 'Iris-versicolor':'x', 'Iris-virginica':'s'}

for species, group in df.groupby('class'):
    axs[0].scatter(group['sepal_length'], group['sepal_width'], color=colors[species], marker=markers[species], label=species)

axs[0].set_title('Sepal Length vs. Sepal Width')
axs[0].set_xlabel('Sepal Length (cm)')
axs[0].set_ylabel('Sepal Width (cm)')
axs[0].legend()

for species, group in df.groupby('class'):
    axs[1].scatter(group['sepal_length'], group['sepal_width'], color=colors[species], marker=markers[species], label=species)

axs[1].set_title('Petal Length vs. Petal Width')
axs[1].set_xlabel('Petal Length (cm)')
axs[1].set_ylabel('Petal Width (cm)')
axs[1].legend()

plt.tight_layout()
plt.savefig('scatter_plot.png')