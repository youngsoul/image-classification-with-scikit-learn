import os
import time
from pathlib import Path
from typing import Literal, List

import numpy as np
import pandas as pd
from embetter.grab import ColumnGrabber
from embetter.multi import ClipEncoder
from embetter.vision import ImageLoader
from sklearn.pipeline import make_pipeline

target_dirs = ['cow', 'elephant', 'horse', 'spider']
root_dir = str(Path(__file__).parent.resolve()) + "/dataset"
target_encodings = {}

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

def grade_submission():
    submission_df = compare_classifications(predicted_csv="zero_shot_test_submission.csv", ground_truth_csv="test_true_values.csv")
    submission_df.to_csv('zero_shot_submission_results.csv', index=False)

    evaluate_classification_results(submission_df)


def create_submission_df(df: pd.DataFrame, class_list: List[str]) -> pd.DataFrame:
    if 'filepath' not in df.columns:
        raise ValueError("DataFrame must contain a 'filepath' column.")
    if len(df) != len(class_list):
        raise ValueError("Length of class list must match number of rows in DataFrame.")

    result_df = pd.DataFrame()
    result_df['ID'] = df['filepath'].apply(lambda x: os.path.basename(x))
    result_df['CLASS'] = class_list
    return result_df

def test_zero_shot_model():
    print("Test Zero Shot model")
    test_files_df = create_filepaths_df(dir_name='Test', dirs=[""])

    image_embedding_pipeline = make_pipeline(
       ColumnGrabber("filepath"),
      ImageLoader(convert="RGB"),
      ClipEncoder(),
    )

    test_X = image_embedding_pipeline.fit_transform(test_files_df)

    y_preds = []
    for i, row in test_files_df.iterrows():
        image_embedding = test_X[i]
        most_similar_target = None
        most_similar_similarity = -1
        for target_dir in target_dirs:
            target_embedding = target_encodings[target_dir]
            similarity = cosine_similarity(image_embedding, target_embedding)
            if similarity > most_similar_similarity:
                most_similar_similarity = similarity
                most_similar_target = target_dir
        y_preds.append(most_similar_target)

    submission_df = create_submission_df(test_files_df, y_preds)
    submission_df.to_csv('zero_shot_test_submission.csv', index=False)

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


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def create_target_encodings(dirs: List = target_dirs) -> dict:
    """Create a dictionary of target names and their encodings."""
    global target_encodings
    for dir in dirs:
        target_encodings[dir] = ClipEncoder().fit_transform(dir)
    return target_encodings


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

def create_test_submission_file():
    # create a test submission from the 'unknown' test dataset
    test_zero_shot_model()

def main():
    training_files_df = create_filepaths_df(dir_name='Train')
    print(f"Training Shape: {training_files_df.shape}")
    # read the Valid filepaths
    validation_files_df = create_filepaths_df(dir_name='Valid')
    print(f"Validation Shape: {validation_files_df.shape}")

    # merge training_files_df and validation_files_df into all_training_df dataframe
    all_training_df = pd.concat([training_files_df, validation_files_df], ignore_index=True)
    print(f"All Training Shape: {all_training_df.shape}")

    # all_training_df = validation_files_df

    print(f"Total training images: {len(all_training_df)}")

    image_embedding_pipeline = make_pipeline(
       ColumnGrabber("filepath"),
      ImageLoader(convert="RGB"),
      ClipEncoder(),
    )

    # convert the filepaths to embeddings
    X = image_embedding_pipeline.fit_transform(all_training_df)
    y = all_training_df['target']
    print(X.shape)
    print(y.shape)

    # for each image embedding, calculate the cosine similarity to the target text embeddings,
    # and compare that with the actual target name
    similarities = []
    for i, row in all_training_df.iterrows():
        image_embedding = X[i]
        target_name = row['target']
        similarity_scores = {}
        most_similar_target = None
        most_similar_similarity = -1
        for target_dir in target_dirs:
            target_embedding = target_encodings[target_dir]
            similarity = cosine_similarity(image_embedding, target_embedding)
            similarity_scores[f"similarity_score_{target_dir}"] = similarity
            if similarity > most_similar_similarity:
                most_similar_similarity = similarity
                most_similar_target = target_dir

        similarity_scores["most_similar_target"] = most_similar_target
        similarity_scores["true_target"] = target_name
        if target_name == most_similar_target:
            similarity_scores["CORRECT"] = 1
        else:
            similarity_scores["CORRECT"] = 0
        # from row, get the filename and parent directory
        filename = os.path.basename(row['filepath'])
        parent_dir = os.path.basename(os.path.dirname(row['filepath']))
        # get the parent directory of the parent_dir
        parent_parent_dir = os.path.basename(os.path.dirname(os.path.dirname(row['filepath'])))
        similarity_scores["filename"] = filename
        similarity_scores["parent_dir"] = parent_parent_dir

        similarities.append(similarity_scores)

    similarities_df = pd.DataFrame(similarities)
    similarities_df.to_csv('zero_shot_similarities.csv', index=False)
    # calculate accuracy of most_similar_target and true_target
    accuracy = similarities_df.apply(lambda row: row['most_similar_target'] == row['true_target'], axis=1).mean()
    print(f"Training / Validation Dataset Accuracy: {accuracy:.2%}")
    print()






if __name__ == '__main__':
    total_s = time.time()

    # create text encodings for the target names
    target_encodings = create_target_encodings()

    main()

    create_test_submission_file()

    grade_submission()

    total_e = time.time()
    print(f"Total time: {total_e-total_s}")

    print("Done")
