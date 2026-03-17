import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures

full_data = pd.read_csv("Stars.csv")

# Plotting the distribution of the 'Type' variable
"""plt.figure(figsize=(8, 5))
full_data['Type'].value_counts().plot(kind='bar', color = ['blue', 'orange', 'green', 'red', 'purple', 'cyan', 'magenta'])
print("Value counts for 'Type':\n", full_data['Type'].value_counts())
plt.title('Distribution of Star Types')
plt.xlabel('Star Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()"""


# Relationship between Spectral Class and Star Type
"""
plt.figure(figsize=(10, 6))
pd.crosstab(full_data['Spectral_Class'], full_data['Type']).plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Spectral Class vs Star Type')
plt.xlabel('Spectral Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Star Type')
plt.show()
"""

#sort x and y by temperature
sorted_indices = np.argsort(full_data['Temperature'].values)
full_data = full_data.iloc[sorted_indices]


#Relatinship between temperature and star type
x = full_data['Temperature'].values.reshape(-1, 1)

print("Temperature shape:", x.shape)

y = full_data['A_M']

poly = PolynomialFeatures(degree=2)

x_poly = poly.fit_transform(x)

print("Polynomial features (first 5 rows):\n", x_poly[:5])
print("Polynomial features shape:", x_poly.shape)
print("A_M shape:", y.shape)

model = LinearRegression()

model.fit(x_poly, y)

pred = model.predict(x_poly)

print("Predicted A_M shape:", pred.shape)
print("Predicted A_M:", pred)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, pred, color='red', label='Linear Fit')
plt.title('Temperature vs Star Type')
plt.xlabel('Temperature')
plt.ylabel('Star Type')
plt.legend()
plt.show()

