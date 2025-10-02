import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
data = pd.DataFrame({
    'size_sqft': [50000, 75000, 100000],
    'insulation_r': [19, 30, 38],
    'hvac_efficiency': [0.8, 0.9, 0.95],
    'savings': [400, 550, 700]
})
X = data.drop('savings', axis=1)
y = data['savings']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(f'MAE: {mean_absolute_error(y_test, preds)}')
new_building = pd.DataFrame({'size_sqft': [80000], 'insulation_r': [25], 'hvac_efficiency': [0.85]})
print(f'Predicted Savings: ${model.predict(new_building)[0]:.0f}/unit')