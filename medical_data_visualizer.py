import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Import the data from medical_examination.csv and assign it to the df variable
df = pd.read_csv('medical_examination.csv')

# Step 2: Create the overweight column in the df variable
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)).apply(lambda x: 1 if x > 25 else 0)

# Step 3: Normalize cholesterol and gluc columns
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

def draw_cat_plot():
    # Step 4: Create a DataFrame for the cat plot using pd.melt
    df_cat = pd.melt(df, id_vars='cardio', value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # Step 5: Group and reformat the data in df_cat to split it by cardio
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # Step 6: Create a chart that shows the value counts of the categorical features
    fig = sns.catplot(x='variable', hue='value', col='cardio', data=df_cat, kind='bar', height=5, aspect=0.7)
    plt.subplots_adjust(top=0.8)
    fig.fig.suptitle('Categorical Plot')
    
    return fig

def draw_heat_map():
    # Step 8: Clean the data in the df_heat variable
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                  (df['height'] >= df['height'].quantile(0.025)) &
                  (df['height'] <= df['height'].quantile(0.975)) &
                  (df['weight'] >= df['weight'].quantile(0.025)) &
                  (df['weight'] <= df['weight'].quantile(0.975))]
    
    # Step 9: Calculate the correlation matrix
    corr = df_heat.corr()
    
    # Step 10: Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Step 11: Set up the matplotlib figure
    plt.figure(figsize=(12, 8))
    
    # Step 12: Plot the correlation matrix using sns.heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    
    plt.title('Heat Map of Correlation Matrix')
    plt.show()

# Do not modify the next two lines
if __name__ == "__main__":
    draw_cat_plot()
    draw_heat_map()
