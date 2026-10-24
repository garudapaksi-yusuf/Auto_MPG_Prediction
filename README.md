# Auto MPG Prediction

The Auto MPG Dataset from UCI Machine Learning Repository provides an excellent foundation for analyzing the key technical factors that influence vehicle fuel efficiency. This project combines feature analysis with statistical prediction: using machine learning models to uncover how vehicle features such as engine displacement, horsepower, vehicle weight, and cylinder count contribute to fuel performance outcomes.

## Project Workflow

The analysis and modeling workflow is documented in the notebooks:

1. [Initial data analysis and cleaning](https://nbviewer.org/github/garudapaksi-yusuf/Auto_MPG_Prediction/blob/main/notebook/1_initial-data-analysis.ipynb)
2. [Exploratory analysis](https://nbviewer.org/github/garudapaksi-yusuf/Auto_MPG_Prediction/blob/main/notebook/2_exploratory-data-analysis.ipynb)
3. [Model development and evaluation](https://nbviewer.org/github/garudapaksi-yusuf/Auto_MPG_Prediction/blob/main/notebook/3_predictive-modeling.ipynb)
4. [Conclusions and recommendations](https://nbviewer.org/github/garudapaksi-yusuf/Auto_MPG_Prediction/blob/main/notebook/4_project-summary.ipynb)

The first notebook fetches dataset from the repository using [ucimlrepo](https://github.com/uci-ml-repo/ucimlrepo), and processed data used by the project is in [data/](https://github.com/garudapaksi-yusuf/Auto_MPG_Prediction/tree/main/data). Project dependencies provided from [requirements.txt](https://github.com/garudapaksi-yusuf/Auto_MPG_Prediction/blob/main/requirements.txt).

## Repository Layout

```text
app/                Streamlit interface and app configuration
data/               Cleaned and transformed datasets
model/              Persisted trained model
notebook/           Analysis, exploration, and modeling notebooks
requirements.txt
```
