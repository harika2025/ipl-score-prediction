# IPL First Innings Score Prediction

This project predicts the final score of the first innings of an IPL match. It looks at the match after the first 5 overs and uses the current score, wickets, overs and the last 5 overs' form to estimate where the innings will finish.

I built this project while learning machine learning, using [IPL-Score-Prediction-with-Machine-Learning](https://github.com/thatfreakcoder/IPL-Score-Prediction-with-Machine-Learning) by thatfreakcoder and the [Kaggle IPL dataset](https://www.kaggle.com/yuvrajdagur/ipl-dataset-season-2008-to-2017) as references.

I then made my own changes:
- Converted the notebook into a single runnable Python script
- Split train and test by match instead of by ball, to avoid leakage
- Replaced the long if/else prediction code with a scikit-learn Pipeline
- Loads the dataset from a link, so nothing needs to be downloaded by hand

## Dataset

Ball-by-ball IPL data from 2008 to 2017, taken from [Kaggle](https://www.kaggle.com/yuvrajdagur/ipl-dataset-season-2008-to-2017). It has 76,014 rows and 15 columns.

## What the code does

- Keeps only the 8 teams that still play in the league
- Removes the first 5 overs of every match
- Drops columns that don't help here (date, venue, batsman, bowler, striker, non-striker)
- One-hot encodes the batting and bowling team
- Splits the data 80/20 by match, so one match never appears in both train and test
- Trains six models and compares them: Decision Tree, Linear Regression, Random Forest, Lasso, SVR and a Neural Network
- Saves the Random Forest as the final model

## Results

In the original notebook (random split by ball), Random Forest scored best with a test R² of about 93%, an average error of about 4.6 runs and an RMSE of about 7.8 runs. Decision Tree reached about 86%, the Neural Network about 85%, and Linear Regression, Lasso and SVR were between 57% and 66%.

Those scores come from a random split, so balls from the same match were in both train and test and the numbers look better than they should. The script in this repo splits by match, so its results will be lower. Run it to see them.

## How to run

```
git clone https://github.com/harika2025/ipl-score-prediction.git
cd ipl-score-prediction
pip install -r requirements.txt
python ipl_score_prediction.py
```

The script uses `data.csv` if it is in the folder. If not, it loads it from this repo's raw link. It prints a results table, saves `results.csv` and `forest_model.pkl`, and shows a few example predictions.

## Limitations

- Trained on 2008 to 2017 data and 8 teams only
- Venue and players are not used as features

## Ideas for later

- Add venue as a feature
- Tune the Random Forest
- Build a small web app

## Built with

Python, pandas, NumPy, scikit-learn, joblib
