from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / 'data' / 'processed' / 'train.csv'
VALIDATION_PATH = PROJECT_ROOT / 'data' / 'processed' / 'validation.csv'
RESULTS_DIR = PROJECT_ROOT / 'results'
FEATURES = ['G', 'MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'PTS']
TARGET_COLUMN = 'MVP'


def validate_dataset(df: pd.DataFrame, dataset_name: str) -> None:
    required = FEATURES + [TARGET_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f'{dataset_name} is missing required columns: {missing}')

    allowed_values = set(df[TARGET_COLUMN].dropna().unique())
    if not allowed_values.issubset({0, 1}):
        raise ValueError(f'{dataset_name} target column must be binary. Found: {sorted(allowed_values)}')

    per_season = df.groupby('Season').size()
    if not per_season.eq(3).all():
        bad = per_season[per_season != 3]
        raise ValueError(f'{dataset_name} must contain exactly three finalists per season. Problem seasons: {bad.to_dict()}')

    winners = df.groupby('Season')[TARGET_COLUMN].sum()
    if not winners.eq(1).all():
        bad = winners[winners != 1]
        raise ValueError(f'{dataset_name} must contain exactly one MVP winner per season. Problem seasons: {bad.to_dict()}')

    if TARGET_COLUMN in FEATURES:
        raise ValueError('Target leakage: target column is included in the feature list.')


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)
    validate_dataset(train_df, 'Training data')
    validate_dataset(validation_df, 'Validation data')

    train_df = train_df[FEATURES + ['Season', 'Player', TARGET_COLUMN]].copy()
    validation_df = validation_df[FEATURES + ['Season', 'Player', TARGET_COLUMN]].copy()
    return train_df, validation_df


def compute_binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }


def save_metrics_csv(metrics: dict[str, float | int], file_name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame({'metric': list(metrics.keys()), 'value': list(metrics.values())})
    metrics_df.to_csv(RESULTS_DIR / file_name, index=False)
    return None


def create_prediction_outputs(
    validation_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    y_prob: pd.Series,
    file_name: str,
) -> pd.DataFrame:
    pred_df = validation_df[['Season', 'Player']].copy()
    pred_df['Actual_MVP'] = y_true.astype(int).to_numpy()
    pred_df['Predicted_MVP'] = y_pred.astype(int).to_numpy()
    pred_df['Predicted_Probability_MVP'] = y_prob.astype(float).to_numpy()
    pred_df = pred_df[['Season', 'Player', 'Actual_MVP', 'Predicted_MVP', 'Predicted_Probability_MVP']]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(RESULTS_DIR / file_name, index=False)
    return pred_df


def create_seasonal_predictions(validation_df: pd.DataFrame, probabilities: np.ndarray, file_name: str) -> pd.DataFrame:
    rows = []
    for season, season_df in validation_df.groupby('Season', sort=True):
        season_df = season_df.sort_values('Player').copy()
        season_probs = probabilities[season_df.index]
        predicted_idx = int(np.argmax(season_probs))
        predicted_player = season_df.iloc[predicted_idx]['Player']
        actual_player = season_df.loc[season_df[TARGET_COLUMN] == 1, 'Player'].iloc[0]
        correct = int(actual_player == predicted_player)

        row = {
            'Season': season,
            'Actual_MVP': actual_player,
            'Predicted_MVP': predicted_player,
            'Correct': correct,
            'Finalist_1_Player': season_df.iloc[0]['Player'],
            'Finalist_1_Probability': float(season_probs[0]),
            'Finalist_2_Player': season_df.iloc[1]['Player'],
            'Finalist_2_Probability': float(season_probs[1]),
            'Finalist_3_Player': season_df.iloc[2]['Player'],
            'Finalist_3_Probability': float(season_probs[2]),
            'Predicted_Probability_MVP': float(season_probs[predicted_idx]),
        }
        rows.append(row)

    seasonal_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seasonal_df.to_csv(RESULTS_DIR / file_name, index=False)
    return seasonal_df


def main() -> None:
    train_df, validation_df = load_data()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET_COLUMN].astype(int)
    X_val = validation_df[FEATURES]
    y_val = validation_df[TARGET_COLUMN].astype(int)

    # Reasonable default hyperparameters; no validation tuning is performed.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = pd.Series(model.predict(X_val), index=validation_df.index)
    y_prob = pd.Series(model.predict_proba(X_val)[:, 1], index=validation_df.index)

    metrics = compute_binary_metrics(y_val, y_pred)
    save_metrics_csv(metrics, 'random_forest_metrics.csv')
    predictions = create_prediction_outputs(validation_df, y_val, y_pred, y_prob, 'random_forest_predictions.csv')
    seasonal_df = create_seasonal_predictions(validation_df, model.predict_proba(X_val)[:, 1], 'random_forest_seasonal_predictions.csv')

    print('Random Forest validation metrics:')
    for key, value in metrics.items():
        print(f'  {key}: {value}')
    print('Prediction rows:', len(predictions))
    print('Seasonal accuracy:', float(seasonal_df['Correct'].mean()))
    print('Saved metrics to:', RESULTS_DIR / 'random_forest_metrics.csv')
    print('Saved predictions to:', RESULTS_DIR / 'random_forest_predictions.csv')
    print('Saved seasonal predictions to:', RESULTS_DIR / 'random_forest_seasonal_predictions.csv')


if __name__ == '__main__':
    main()
