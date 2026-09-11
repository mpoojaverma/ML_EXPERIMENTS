# Experiment 3 – Linear Regression

## Aim

To implement Linear Regression to perform prediction.

## Dataset

**Flood Risk Dataset – India**

The dataset contains 10,000 records and 14 attributes.

## Objective

The experiment uses:

- Independent Variable: `River Discharge (m³/s)`
- Dependent Variable: `Water Level (m)`

The objective is to predict the water level based on river discharge.

## Algorithm

1. Load the flood risk dataset.
2. Select river discharge as the independent variable.
3. Select water level as the dependent variable.
4. Reshape the input variable into a two-dimensional array.
5. Split the dataset into training and testing sets.
6. Implement Linear Regression using the Ordinary Least Squares Normal Equation.
7. Calculate the intercept and regression coefficient.
8. Predict the water level for the test data.
9. Evaluate the model using Mean Squared Error (MSE).
10. Evaluate the model using the R² score.
11. Visualize the actual test data and regression line.

## Method

The regression coefficients are calculated using the Normal Equation:

θ = (XᵀX)⁻¹XᵀY

## Evaluation Metrics

### Mean Squared Error (MSE)

Measures the average squared difference between actual and predicted values.

### R² Score

Measures how well the regression model explains the variation in the dependent variable.

## Visualization

The experiment generates a scatter plot of the actual test observations along with the fitted linear regression line.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Files

- `exp_3.py` – Python implementation
- `flood_risk_dataset_india.csv` – Dataset used
- `output/linear_regression.png` – Regression visualization
- `output/output.txt` – Terminal output

## Result

Thus, the implementation of Linear Regression to perform prediction was successfully executed.