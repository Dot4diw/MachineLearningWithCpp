import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("moons_training_data.csv",header=None)

#df = pd.read_csv("moons_data.csv",header=None)

df.columns = ['x','y','category']

# append the new data.
df.loc[len(df.index)] = [1.8842326, 0.050685, 'new']

sns.scatterplot(data=df, x='x', y='y',hue='category',
                 palette=["blue","red","yellow"])
plt.savefig('new_data_predicted_result.pdf')
plt.show()


#######################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

x, y = make_moons(n_samples=2000, shuffle=True,
                 noise=0.08, random_state=None)

df = pd.DataFrame(x)
df[2] = pd.DataFrame(y)
df.columns = ['x','y','category']

df.to_csv("moons_data.csv", index=False, header = False)
sns.scatterplot(data=df, x='x', y='y',hue='category',
                palette=["blue","red","yellow"])
plt.savefig('moons_data.pdf')
plt.show()
