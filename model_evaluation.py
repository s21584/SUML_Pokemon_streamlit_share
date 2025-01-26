import pandas as pd
from sentence_transformers import SentenceTransformer, util
from model_api import invoke_batch


def evaluate_model():
    """
        Evaluates the performance of a text generation model by computing the cosine similarity
        between generated descriptions and their corresponding reference descriptions.

        The function uses the SentenceTransformer model to encode the reference and generated
        descriptions into embeddings and calculates the cosine similarity between them. It saves
        the results (including references, predictions, and similarity scores) into a CSV file
        named 'evaluations.csv' and prints the mean similarity as a percentage.

        Steps:
        1. Fetches test data using the `invoke_batch` function.
        2. Encodes reference and prediction strings into embeddings.
        3. Computes cosine similarity for each reference-prediction pair.
        4. Stores the results in a DataFrame and exports it to a CSV file.
        5. Prints the mean similarity percentage.

        Returns:
            None
    """

    model = SentenceTransformer('all-MiniLM-L6-v2')

    test_data = invoke_batch("RidzIn/Pokemon-Describer", "data/test.csv")

    results = {"y_true": [], "y_pred": [], "similarity": []}

    for row in range(len(test_data['generated_descriptions'])):
        reference = test_data["originals"][row]
        prediction = test_data["generated_descriptions"][row]

        ref_embedding = model.encode(reference, convert_to_tensor=True)
        pred_embedding = model.encode(prediction, convert_to_tensor=True)

        similarity = util.cos_sim(ref_embedding, pred_embedding)

        results['y_true'].append(reference)
        results['y_pred'].append(prediction)
        results['similarity'].append(similarity.item())

    df = pd.DataFrame(results)
    df.to_csv("evaluations.csv", index=False)

    print(f"Mean similarity metric: {round(df['similarity'].mean()*100, 2)}%")


if __name__ == "__main__":
    evaluate_model()

