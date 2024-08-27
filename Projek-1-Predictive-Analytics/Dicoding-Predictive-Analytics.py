#%%
import opendatasets as od
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
#%%
od.download('https://www.kaggle.com/datasets/ayushparwal2026/cars-dataset')
#%% md
# # Get Dataset Information 
# 
# this section below are useful to finding out the information related to the dataset. use `.info()` and `.describe()`
#%%
data = pd.read_csv('cars-dataset/used_cars_data.csv')
data.head()
#%%
data.info()
#%%
data.describe()
#%% md
# # Determine the columns to be used as feature and label
# 
# in this case i want to predict the fuel consumption per vehicle, so i the columns i need are year, fuel type, transmission, mileage, engine, power, and seats.
# 
# the label is the mileage, from the data, mileage column contain the fuel consumption. but there are 3 types of engine, in Indonesia, a family vehicle are only in 2 options, diesel or petrol. 
#%%
df = data[['Year', 'Fuel_Type', 'Transmission', 'Mileage', 'Engine', 'Power', 'Seats']]
#%%
df.head()
#%% md
# after we determine the columns to be used, then we must remove the excess data, like the fuel type, i want to use only diesel and petrol to be predicted.
#%%
df = df[df['Fuel_Type'].isin(['Petrol', 'Diesel'])]
df.head()
#%%
df.Fuel_Type.unique()
#%% md
# let's display the variable correlation
# 
# before we see the corrlation, we need to change all the dtype to numeric. for categorical we must map them
# 
# Diesel = 0<br>
# Petrol = 1
# 
# Manual = 0<br>
# Automatic = 1
#%%
for_corr = df.copy()
#%%
# mapping
fuel_map = lambda x: 1 if x.lower() == 'petrol' else 0
trans_map = lambda x: 1 if x.lower() == 'automatic' else 0
#%%
for_corr.Fuel_Type = for_corr.Fuel_Type.map(fuel_map)
for_corr.Transmission = for_corr.Transmission.map(trans_map)
#%% md
# lets change the the rest by remove all the string from the engine, mileage, and power. After that we cast the dtype to numeric(float)
#%%
# extractor 
ext = lambda x: x.replace('null bhp', np.NaN).str.replace(r'[^0-9.]', '', regex=True).astype(np.float64)
#%%
for_corr[['Engine', 'Mileage', 'Power']] = for_corr[['Engine', 'Mileage', 'Power']].apply(ext)
#%%
for_corr.info()
#%%
for_corr.head()
#%%
corr_matrix = for_corr.drop('Mileage', axis=1).corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')

plt.title('Correlation Matrix')
plt.show()
#%% md
# as we can see in the correlation heatmap, Year of a car doesn't have much effect on the data. so we can drop 'Year' column, and then we can see the Seats, seats has much impact on the engine, because in reality, the 5 seater car has smaller weight than 7 seater
#%%
df = df.drop('Year', axis=1)
df.head()
#%% md
# Let's find the missing value first and then re apply the preprocessing function that we have created before
#%%
df.isna().sum()
#%%
df.count()
#%%
df = df.dropna()
df.count()
#%%
df.Fuel_Type = df.Fuel_Type.map(fuel_map)
df.Transmission = df.Transmission.map(trans_map)
df[['Engine', 'Mileage', 'Power']] = df[['Engine', 'Mileage', 'Power']].apply(ext)
df.head()
#%% md
# ## Note
# 
# the prediction on car fuel consumption is close to regression.<br>
# Because of that we need to sort the Mileage column from lowest to highest 
#%%
sorted_data = df.sort_values(by='Mileage', ascending=True)
sorted_data.head(10)
#%%
sorted_data.isna().sum()
#%%
(sorted_data['Mileage'] == 0).sum()
#%%
# deleting variable to free up memory usage
del sorted_data, for_corr
#%% md
# # Removing unnecessary data
# 
# After the data is cleaning, there are still some NaN value in DataFrame, this because the actual data from the dataset are like 'null bhp' and the mileage are 0.
# 
# Both of the criteria must be dropped to clean the data<br>
# 1. Remove NaN
# 2. Remove 0 on Mileage
#%%
df = df.dropna()
df.head(10)
#%%
df = df[df['Mileage'] != 0]
df.head(10)
#%%
df.info()
#%% md
# ## Plot for the relation each Variable with Mileage
# 
# using the scatter plot for easier to read<br> 
# 1. Engine to Mileage
# 2. Power to Mileage
# 3. Seats to Mileage
#%%
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Engine', y='Mileage', data=df)
plt.title('Scatter Plot of Engine vs Mileage')
plt.xlabel('Engine')
plt.ylabel('Mileage')
plt.show()
#%%
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Power', y='Mileage', data=df)
plt.title('Scatter Plot of Power vs Mileage')
plt.xlabel('Engine')
plt.ylabel('Mileage')
plt.show()
#%%
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Seats', y='Mileage', data=df)
plt.title('Scatter Plot of Seats vs Mileage')
plt.xlabel('Engine')
plt.ylabel('Mileage')
plt.show()
#%%
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Transmission', y='Mileage', data=df)
plt.title('Scatter Plot of Transmission vs Mileage')
plt.xlabel('Engine')
plt.ylabel('Mileage')
plt.show()
#%% md
# # Split for train and test also the feature and label
# 
# using `train_test_split` from scikit learn to help create a test batch with last 20% of data. 
#%%
df.head()
#%%
x, y = df.drop('Mileage', axis=1), df[['Mileage']]
#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, shuffle=False, random_state=102)
x_train.head()
#%%
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5, )),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1),
])

model.summary()
#%%
tf.keras.utils.plot_model(model, show_shapes=True)
#%%
model.compile(
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.Adam()
)
#%%
class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_loss') <= 3:
            print('\n\nMSE loss under than equal 3, so stop training')
            self.model.stop_training = True
#%%
H = model.fit(
    x_train, y_train,
    epochs=1000,
    validation_split=.2,
    verbose=2,
    callbacks=[MyCustomCallback()]
)
#%%
plt.figure(figsize=(10, 6))
plt.plot(H.history['loss'], label='Training Loss')
plt.plot(H.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
#%%
prediction = model.predict(x_test)
#%%
prediction
#%%
def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred)).numpy()

def mean_absolute_percentage_error(y_true, y_pred):
    if np.any(y_true == 0):
        raise ValueError("MAPE cannot be calculated when actual values contain zero.")
    
    absolute_percentage_errors = np.abs((y_true - y_pred) / y_true) * 100
    return np.mean(absolute_percentage_errors)
#%%
print(f"test MSE : {mean_squared_error(y_test, prediction)}")
#%%
print(f"test MAPE : {mean_absolute_percentage_error(y_test, prediction)}")
#%% md
# ### Make a dummy test
#%%
dummy = pd.DataFrame({
    'fuel': ['Petrol'],
    'transmission': ['Manual'],
    'engine': ['6200'],
    'power': ['650'],
    'seats': ['5']
})

dummy.fuel = dummy.fuel.map(fuel_map) 
dummy.transmission = dummy.transmission.map(fuel_map)
dummy[['engine', 'power', 'seats']] = dummy[['engine', 'power', 'seats']].apply(ext)
dummy

#%%
model.predict(dummy)
#%%
