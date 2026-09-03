# Logistics Delivery Analysis — Preprocessing Results

## Dataset Summary

| Item | Result |
|---|---:|
| Original dataset rows | 25,000 |
| Original dataset columns | 15 |
| Final dataset rows | 25,000 |
| Final dataset columns | 16 |
| Initial duplicate delivery IDs detected | 498 |
| Final missing values | 0 |
| Final exact duplicate rows | 0 |

## Data Quality Checks

The dataset was checked for missing values, duplicate records, incorrect data types, and unusual values.

- Missing values were handled so that the final dataset contains zero missing values.
- Duplicate delivery IDs were identified during the initial data-quality check. There were 498 duplicate delivery IDs.
- Exact duplicate rows were removed or resolved. The cleaned dataset contains zero exact duplicate rows.
- Data types were reviewed and corrected where necessary.
- A new column was included during preprocessing, increasing the number of columns from 15 to 16.

## Outlier Detection Using IQR

The Interquartile Range (IQR) method was used to identify potential outliers in numerical delivery variables.

Formula used:

```text
IQR = Q3 − Q1
Lower Bound = Q1 − 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Potential outliers were reviewed for the following variables:

- Delivery cost
- Delivery time in hours
- Distance
- Package weight

### Delivery Time Outliers

Using the IQR method, **203 potential outliers** were detected in the `delivery_time_hours` column.

These records were retained for review because unusually long or short delivery times may represent genuine operational situations, such as traffic delays, weather conditions, remote delivery locations, or urgent deliveries.

Boxplots for the numerical variables are available in:

```text
outputs/charts/
```

## Normalization

Numerical features were normalized to place them on comparable scales before further analysis or modelling.

Min-Max normalization was used:

```text
Normalized Value = (Value − Minimum Value) / (Maximum Value − Minimum Value)
```

Normalization supports fair comparison between variables with different units and ranges, such as distance, package weight, delivery cost, and delivery time.

## Final Validation

The cleaned dataset was validated after preprocessing.

| Validation Check | Final Result |
|---|---:|
| Missing values | 0 |
| Exact duplicate rows | 0 |
| Dataset rows retained | 25,000 |
| Final number of columns | 16 |
| Delivery-time potential outliers identified | 203 |

## Conclusion

The logistics delivery dataset was successfully cleaned and prepared for analysis. The final dataset is complete, has no missing values or exact duplicate rows, includes the required preprocessing updates, and is ready for exploratory analysis, visualization, and modelling.