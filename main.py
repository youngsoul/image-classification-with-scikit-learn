from embetter.vision import ImageLoader
from embetter.multi import ClipEncoder
from embetter.grab import ColumnGrabber

from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

import pandas as pd
from pathlib import Path
from typing import Literal, List
import time
import os


target_dirs = ['cow', 'elephant', 'horse', 'spider']
root_dir = str(Path(__file__).parent.resolve()) + "/dataset"

def evaluate_classification_results(submission_results_df: pd.DataFrame) -> None:
    if not {'CLASS', 'TRUE_CLASS', 'CORRECT'}.issubset(submission_results_df.columns):
        raise ValueError("DataFrame must contain 'CLASS', 'TRUE_CLASS', and 'CORRECT' columns.")

    # Accuracy
    accuracy = submission_results_df['CORRECT'].mean()
    print(f"Submission Accuracy: {accuracy:.2%}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    confusion_matrix = pd.crosstab(
        submission_results_df['TRUE_CLASS'],
        submission_results_df['CLASS'],
        rownames=['Actual'],
        colnames=['Predicted'],
        dropna=False
    )
    print(confusion_matrix)


def compare_classifications(predicted_csv: str, ground_truth_csv: str) -> pd.DataFrame:
    # Read both CSV files
    predicted_df = pd.read_csv(predicted_csv)
    ground_truth_df = pd.read_csv(ground_truth_csv)

    # Rename the CLASS column in the second dataframe
    ground_truth_df = ground_truth_df.rename(columns={'CLASS': 'TRUE_CLASS'})

    # Merge on the ID column
    merged_df = pd.merge(predicted_df, ground_truth_df, on='ID')

    # Create CORRECT column: 1 if CLASS = TRUE_CLASS, else 0
    merged_df['CORRECT'] = (merged_df['CLASS'] == merged_df['TRUE_CLASS']).astype(int)

    return merged_df


def create_submission_csv(df: pd.DataFrame, class_list: List[str]) -> pd.DataFrame:
    if 'filepath' not in df.columns:
        raise ValueError("DataFrame must contain a 'filepath' column.")
    if len(df) != len(class_list):
        raise ValueError("Length of class list must match number of rows in DataFrame.")

    result_df = pd.DataFrame()
    result_df['ID'] = df['filepath'].apply(lambda x: os.path.basename(x))
    result_df['CLASS'] = class_list
    return result_df

def create_filepaths_df(dir_name: Literal['Train','Valid'], dirs: List = target_dirs) -> pd.DataFrame:
    data = []
    for dir in dirs:
        for file in Path(f'{root_dir}/{dir_name}/{dir}').glob('*.jpg'):
            row_data = {
                'filepath': file,
                'target': dir
            }
            data.append(row_data)
    files_df = pd.DataFrame(data, columns=["filepath", "target"])
    return files_df



def train() -> LogisticRegression:
    print("Training the model")
    training_files_df = create_filepaths_df(dir_name='Train')

    # create pipeline to read the filepath column, load the image, and encode the image
    image_embedding_pipeline = make_pipeline(
       ColumnGrabber("filepath"),
      ImageLoader(convert="RGB"),
      ClipEncoder(),
    )

    # convert the filepaths to embeddings
    X = image_embedding_pipeline.fit_transform(training_files_df)
    y = training_files_df['target']
    print(X.shape)
    print(y.shape)

    # create a baseline, default model
    model = LogisticRegression(solver='liblinear', max_iter=1_000)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=cv)
    print(scores)
    print(f"Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")

    # train the model on all of the data
    model = LogisticRegression(solver='liblinear', max_iter=1_000)
    model.fit(X, y)

    return model

def validate(model: LogisticRegression):
    print("Validating the model")
    validation_files_df = create_filepaths_df(dir_name='Valid')

    # create pipeline to read the filepath column, load the image, and encode the image
    image_embedding_pipeline = make_pipeline(
       ColumnGrabber("filepath"),
      ImageLoader(convert="RGB"),
      ClipEncoder(),
    )

    validation_X = image_embedding_pipeline.fit_transform(validation_files_df)
    validation_y = validation_files_df['target']

    y_pred = model.predict(validation_X)
    print(accuracy_score(validation_y, y_pred))

def test_model(model: LogisticRegression):
    print("Testing the model")
    test_files_df = create_filepaths_df(dir_name='Test', dirs=[""])

    image_embedding_pipeline = make_pipeline(
       ColumnGrabber("filepath"),
      ImageLoader(convert="RGB"),
      ClipEncoder(),
    )

    test_X = image_embedding_pipeline.fit_transform(test_files_df)

    y_pred = model.predict(test_X)

    submission_df = create_submission_csv(test_files_df, y_pred)
    submission_df.to_csv('test_submission.csv', index=False)


def grade_submission():
    submission_df = compare_classifications(predicted_csv="test_submission.csv", ground_truth_csv="test_true_values.csv")
    submission_df.to_csv('submission_results.csv', index=False)

    evaluate_classification_results(submission_df)



if __name__ == "__main__":
    total_s = time.time()
    s = time.time()
    model = train()
    e = time.time()
    print(f"Training time: {e-s}")

    s = time.time()
    validate(model)
    e = time.time()
    print(f"Validation time: {e-s}")

    s = time.time()
    test_model(model)
    e = time.time()
    print(f"Testing time: {e-s}")

    grade_submission()

    total_e = time.time()
    print(f"Total time: {total_e-total_s}")