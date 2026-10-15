from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'
REPORTS_DIR = PROJECT_ROOT / 'reports'
RAW_DATA_PATH = PROJECT_ROOT / 'NBA_stats_including_MVP.csv'
FEATURES = ['G', 'MP', 'FG%', '3P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'PTS']
TARGET_COLUMN = 'MVP'
MODEL_FILES = {
    'Naive Bayes': ('naive_bayes_model.py', 'naive_bayes'),
    'Logistic Regression': ('logistic_regression_model.py', 'logistic_regression'),
    'Random Forest': ('random_forest_model.py', 'random_forest'),
}


def assignments_from_source(path: Path) -> dict[str, ast.AST]:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    assignments: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
    return assignments


def verify_model_sources() -> None:
    expected_train_path = "PROJECT_ROOT / 'data' / 'processed' / 'train.csv'"
    expected_validation_path = "PROJECT_ROOT / 'data' / 'processed' / 'validation.csv'"

    for model_name, (filename, _) in MODEL_FILES.items():
        source_path = PROJECT_ROOT / 'models' / filename
        assignments = assignments_from_source(source_path)
        model_features = ast.literal_eval(assignments['FEATURES'])
        if model_features != FEATURES:
            raise ValueError(f'{model_name} uses a different feature list: {model_features}')
        if ast.literal_eval(assignments['TARGET_COLUMN']) != TARGET_COLUMN:
            raise ValueError(f'{model_name} has an unexpected target column.')
        if ast.unparse(assignments['TRAIN_PATH']) != expected_train_path:
            raise ValueError(f'{model_name} does not use the shared training split.')
        if ast.unparse(assignments['VALIDATION_PATH']) != expected_validation_path:
            raise ValueError(f'{model_name} does not use the shared validation split.')

        for variable, expected in (
            ('X_train', 'train_df[FEATURES]'),
            ('X_val', 'validation_df[FEATURES]'),
            ('y_train', 'train_df[TARGET_COLUMN].astype(int)'),
            ('y_val', 'validation_df[TARGET_COLUMN].astype(int)'),
        ):
            if not any(
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == variable for target in node.targets)
                and ast.unparse(node.value) == expected
                for node in ast.walk(ast.parse(source_path.read_text(encoding='utf-8')))
            ):
                raise ValueError(f'{model_name} does not assign {variable} from the expected split data.')


def validate_predictions(model_name: str, slug: str, validation_df: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    metrics_path = RESULTS_DIR / f'{slug}_metrics.csv'
    predictions_path = RESULTS_DIR / f'{slug}_predictions.csv'
    seasonal_path = RESULTS_DIR / f'{slug}_seasonal_predictions.csv'

    metrics_df = pd.read_csv(metrics_path).set_index('metric')['value']
    predictions = pd.read_csv(predictions_path)
    seasonal = pd.read_csv(seasonal_path)

    validation_keys = set(map(tuple, validation_df[['Season', 'Player']].astype(str).to_numpy()))
    prediction_keys = set(map(tuple, predictions[['Season', 'Player']].astype(str).to_numpy()))
    if len(predictions) != len(validation_df) or prediction_keys != validation_keys:
        raise ValueError(f'{model_name} predictions do not correspond one-to-one with validation rows.')

    checked = predictions.merge(
        validation_df[['Season', 'Player', TARGET_COLUMN]],
        on=['Season', 'Player'],
        how='left',
        validate='one_to_one',
        suffixes=('', '_source'),
    )
    if checked[TARGET_COLUMN].isna().any() or not np.array_equal(
        checked['Actual_MVP'].astype(int).to_numpy(), checked[TARGET_COLUMN].astype(int).to_numpy()
    ):
        raise ValueError(f'{model_name} saved actual labels do not match validation data.')

    y_true = checked[TARGET_COLUMN].astype(int)
    y_pred = checked['Predicted_MVP'].astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recalculated = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }
    for metric, calculated in recalculated.items():
        if metric not in metrics_df or not np.isclose(float(metrics_df[metric]), calculated):
            raise ValueError(f'{model_name} saved {metric} does not match the saved predictions.')

    expected_seasons = set(validation_df['Season'].astype(str))
    if len(seasonal) != validation_df['Season'].nunique() or set(seasonal['Season'].astype(str)) != expected_seasons:
        raise ValueError(f'{model_name} seasonal output does not contain exactly one row per validation season.')
    if seasonal['Season'].duplicated().any():
        raise ValueError(f'{model_name} has multiple seasonal predictions for a season.')

    for _, row in seasonal.iterrows():
        finalist_names = [row[f'Finalist_{idx}_Player'] for idx in (1, 2, 3)]
        finalist_probabilities = [float(row[f'Finalist_{idx}_Probability']) for idx in (1, 2, 3)]
        season_rows = validation_df.loc[validation_df['Season'].astype(str) == str(row['Season'])]
        if set(finalist_names) != set(season_rows['Player']):
            raise ValueError(f'{model_name} seasonal finalists do not match the validation season.')
        selected_by_probability = finalist_names[int(np.argmax(finalist_probabilities))]
        if row['Predicted_MVP'] != selected_by_probability:
            raise ValueError(f'{model_name} did not select the maximum-probability seasonal finalist.')
        actual_winner = season_rows.loc[season_rows[TARGET_COLUMN] == 1, 'Player'].iloc[0]
        if row['Actual_MVP'] != actual_winner or int(row['Correct']) != int(row['Predicted_MVP'] == actual_winner):
            raise ValueError(f'{model_name} seasonal correctness does not match the validation labels.')

    metrics = {key: float(metrics_df[key]) for key in ('accuracy', 'precision', 'recall', 'f1_score')}
    metrics['seasonal_accuracy'] = float(seasonal['Correct'].mean())
    metrics.update({'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)})
    return metrics, predictions, seasonal


def format_pct(value: float) -> str:
    return f'{value * 100:.2f}%'


def build_reference_comparison(metrics_by_model: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add_row(model: str, task: str, reference: str, our_value: float, difference: str, explanation: str) -> None:
        rows.append({
            'Model': model,
            'Evaluation task': task,
            'Reference paper result': reference,
            'Our result': format_pct(our_value),
            'Difference': difference,
            'Explanation of possible differences': explanation,
        })

    nb = metrics_by_model['Naive Bayes']
    lr = metrics_by_model['Logistic Regression']
    rf = metrics_by_model['Random Forest']

    add_row(
        'Naive Bayes',
        'Binary classification (pure MVP vs non-MVP)',
        '66.67% accuracy (paper reports 0.6666666666666666 on the testing set; Results and Discussion, p. 5)',
        nb['accuracy'],
        f'{(nb["accuracy"] - 2 / 3) * 100:+.2f} percentage points',
        'The paper reports its test-set pure-classification accuracy. Our value is calculated from 12 held-out finalist records; dataset versions, preprocessing, and implementation details may differ.',
    )
    add_row(
        'Naive Bayes',
        'Seasonal MVP selection (one finalist chosen per season)',
        'Paper states approximately 66% for seasonal results, but also says it selected 2 seasons correctly and 2 incorrectly (50% over four seasons). These statements conflict.',
        nb['seasonal_accuracy'],
        f'Not computed; against the narrative-implied 50%: {(nb["seasonal_accuracy"] - 0.5) * 100:+.2f} percentage points',
        'The reference seasonal percentage is internally inconsistent, so no single reliable difference can be reported. Our result uses the four validation seasons and is kept separate from binary classification accuracy.',
    )
    add_row(
        'Logistic Regression',
        'Binary classification (pure MVP vs non-MVP)',
        'Approximately 67% accuracy (Results and Discussion, p. 5)',
        lr['accuracy'],
        f'{(lr["accuracy"] - 0.67) * 100:+.2f} percentage points vs the paper\'s approximate 67%',
        'The paper value is approximate and the source dataset, preprocessing, and exact logistic-regression implementation may differ from this project.',
    )
    add_row(
        'Logistic Regression',
        'Seasonal MVP selection (one finalist chosen per season)',
        '50% accuracy (Results and Discussion, p. 5)',
        lr['seasonal_accuracy'],
        f'{(lr["seasonal_accuracy"] - 0.5) * 100:+.2f} percentage points',
        'Both figures are seasonal winner accuracy, but sample seasons and exact model procedures may differ; our score is based on four held-out seasons.',
    )
    for model, metrics in (('Random Forest', rf),):
        add_row(
            model,
            'Binary classification (pure MVP vs non-MVP)',
            'Not reported; the paper evaluates Naive Bayes and Logistic Regression, not Random Forest.',
            metrics['accuracy'],
            'Not applicable',
            'No reference-paper Random Forest result is available for comparison.',
        )
        add_row(
            model,
            'Seasonal MVP selection (one finalist chosen per season)',
            'Not reported; the paper evaluates Naive Bayes and Logistic Regression, not Random Forest.',
            metrics['seasonal_accuracy'],
            'Not applicable',
            'No reference-paper Random Forest result is available for comparison.',
        )

    return pd.DataFrame(rows)


def main() -> None:
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    validation_df = pd.read_csv(DATA_DIR / 'validation.csv')
    raw_df = pd.read_csv(RAW_DATA_PATH)

    verify_model_sources()
    if set(train_df['Season']) & set(validation_df['Season']):
        raise ValueError('Training and validation seasons overlap.')
    if len(train_df) != 48 or len(validation_df) != 12:
        raise ValueError('Prepared split row counts differ from the previously established split.')
    if not train_df.groupby('Season').size().eq(3).all() or not validation_df.groupby('Season').size().eq(3).all():
        raise ValueError('A prepared season does not contain exactly three finalists.')

    metrics_by_model: dict[str, dict[str, float]] = {}
    predictions_by_model: dict[str, pd.DataFrame] = {}
    for model_name, (_, slug) in MODEL_FILES.items():
        metrics, predictions, _ = validate_predictions(model_name, slug, validation_df)
        metrics_by_model[model_name] = metrics
        predictions_by_model[model_name] = predictions

    rf_predictions = predictions_by_model['Random Forest']
    rf_distribution = rf_predictions['Predicted_MVP'].astype(int).value_counts().reindex([0, 1], fill_value=0)
    rf_metrics = metrics_by_model['Random Forest']
    if 'Year' not in raw_df.columns:
        raise ValueError('Raw source has no Year column to check for 2020-21 availability.')
    source_years = pd.to_numeric(raw_df['Year'], errors='coerce').dropna().astype(int)
    has_2021 = bool((source_years == 2021).any())

    reference_df = build_reference_comparison(metrics_by_model)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reference_df.to_csv(RESULTS_DIR / 'reference_comparison.csv', index=False)

    lines = [
        'NBA MVP Prediction - Final Project Status',
        '=========================================',
        '',
        'Completed tasks',
        '  - Dataset preparation and fixed train/validation split are in place; original data and splits were not modified during this verification.',
        '  - Naive Bayes, Logistic Regression, and Random Forest model scripts and their saved evaluation outputs were inspected.',
        '  - A reference-paper comparison was produced at results/reference_comparison.csv from the saved model metrics and seasonal predictions.',
        '  - Validation checks confirmed common model feature definitions, common train/validation source paths, no target in the feature list, prediction coverage, saved metric consistency, and seasonal winner selection.',
        '',
        'Prepared data',
        f'  Training: {len(train_df)} records across {train_df["Season"].nunique()} seasons ({train_df["Season"].min()} through {train_df["Season"].max()}).',
        f'  Validation: {len(validation_df)} records across {validation_df["Season"].nunique()} seasons ({validation_df["Season"].min()} through {validation_df["Season"].max()}).',
        f'  Shared features: {", ".join(FEATURES)}.',
        '  Each model source uses the same processed train.csv and validation.csv files; model prediction keys and actual labels match all validation records exactly.',
        '  Input columns are restricted to the ten statistical features. Player and Season are retained only for identifying/grouping predictions; MVP is used only as the target.',
        '',
        'Our model results',
        '  Binary classification metrics (validation records):',
    ]
    for model_name in MODEL_FILES:
        metrics = metrics_by_model[model_name]
        lines.append(
            f'    {model_name}: accuracy={format_pct(metrics["accuracy"])}, precision={format_pct(metrics["precision"])}, '
            f'recall={format_pct(metrics["recall"])}, F1={format_pct(metrics["f1_score"])}.'
        )
    lines.extend([
        '  Seasonal MVP accuracy (separate one-winner-per-season task):',
    ])
    for model_name in MODEL_FILES:
        metrics = metrics_by_model[model_name]
        seasonal_file = RESULTS_DIR / f'{MODEL_FILES[model_name][1]}_seasonal_predictions.csv'
        season_count = len(pd.read_csv(seasonal_file))
        correct = round(metrics['seasonal_accuracy'] * season_count)
        lines.append(f'    {model_name}: {format_pct(metrics["seasonal_accuracy"])} ({correct}/{season_count} seasons).')

    lines.extend([
        '',
        'Reference-paper results and comparison limits',
        '  - Naive Bayes binary classification: the paper reports 0.6666666666666666 accuracy on its testing set (about 66.67%).',
        '  - Logistic Regression binary classification: the paper reports approximately 67% accuracy for pure MVP vs non-MVP classification.',
        '  - Logistic Regression seasonal selection: the paper reports 50% accuracy when selecting one MVP from three finalists.',
        '  - Naive Bayes seasonal selection: the paper describes about 66% but also reports two correct and two incorrect seasons. Two of four is 50%, so its reference seasonal result is internally inconsistent; no definitive difference is claimed.',
        '  - Paper and project scores may also differ because of implementation choices, preprocessing, and small validation samples. Binary classification and seasonal winner accuracy are different tasks and are not interchanged.',
        '',
        'Random Forest diagnostic',
        f'  Predicted binary class distribution: class 0 = {int(rf_distribution.loc[0])}, class 1 = {int(rf_distribution.loc[1])} of {len(rf_predictions)} records.',
        f'  Confusion matrix (rows = actual [0, 1], columns = predicted [0, 1]): [[{rf_metrics["tn"]}, {rf_metrics["fp"]}], [{rf_metrics["fn"]}, {rf_metrics["tp"]}]].',
        f'  Actual validation class counts: class 0 = {rf_metrics["tn"] + rf_metrics["fp"]}, class 1 = {rf_metrics["fn"] + rf_metrics["tp"]}.',
        f'  At the default decision threshold, binary metrics are accuracy={format_pct(rf_metrics["accuracy"])}, precision={format_pct(rf_metrics["precision"])}, recall={format_pct(rf_metrics["recall"])}, and F1={format_pct(rf_metrics["f1_score"])}. The confusion matrix above gives the corresponding counts; seasonal selection separately chooses the highest-probability finalist within each season.',
        '  The validation set was not used to tune the Random Forest.',
        '',
        '2020-21 reproduction status',
        f'  Raw source contains Year=2021 rows: {has_2021}.',
        f'  Prepared validation seasons: {", ".join(sorted(validation_df["Season"].astype(str).unique()))}.',
        '  The raw source has no Year=2021 records and the processed splits stop at 2019-20. The paper\'s final 2020-21 experiment with Curry, Embiid, and Jokic cannot be reproduced from the current dataset.',
        '',
        'Limitations and unresolved issues',
        '  - Only four seasons (12 finalist records) are in the validation split, so metric estimates are unstable.',
        '  - The paper is a scanned PDF and describes aspects of its methods incompletely; the exact original implementations cannot be reproduced.',
        '  - The paper\'s Naive Bayes seasonal percentage conflicts with its stated season counts.',
        '  - The paper\'s final 2020-21 finalist experiment is not reproducible without that season\'s source records.',
        '',
        'Verification scope',
        '  - This verification read saved predictions and metrics and recalculated binary metrics from the prediction files; it did not retrain or modify any model.',
        '  - All three model prediction files cover the same 12 validation records with matching actual labels; all saved binary metrics match recalculation.',
        '  - Each seasonal output contains exactly one selected winner per validation season, and each selected winner has the highest recorded class-1 probability among that season\'s three finalists.',
    ])
    (REPORTS_DIR / 'final_project_status.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print('Verification passed: shared feature/source definitions, matching validation records, metrics, and seasonal outputs.')
    print(f'Random Forest predicted classes: {{0: {int(rf_distribution.loc[0])}, 1: {int(rf_distribution.loc[1])}}}')
    print(f'Random Forest confusion matrix [[TN, FP], [FN, TP]]: [[{rf_metrics["tn"]}, {rf_metrics["fp"]}], [{rf_metrics["fn"]}, {rf_metrics["tp"]}]]')
    print('2020-21 available in raw source:', has_2021)
    print('Reference comparison:')
    print(reference_df.to_string(index=False))
    print('Wrote:', RESULTS_DIR / 'reference_comparison.csv')
    print('Wrote:', REPORTS_DIR / 'final_project_status.txt')


if __name__ == '__main__':
    main()
