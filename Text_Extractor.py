import json
import requests
import numpy as np
from scipy.spatial.distance import cosine


def get_embedding(text, model="bge-m3:latest"):
    """Get embeddings from Ollama for the user query"""
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": model,
            "prompt": text
        }
    )
    if response.status_code == 200:
        return response.json()["embedding"]
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return 1 - cosine(vec1, vec2)


def find_closest_embeddings(user_text, embeddings_file, top_n=3):
    """Find the top N closest embeddings to the user text"""
    # Get embedding for user text
    user_embedding = get_embedding(user_text)
    if not user_embedding:
        return []

    # Load embeddings from JSON file
    with open(embeddings_file, 'r') as f:
        embeddings_data = json.load(f)

    # Calculate similarities across all categories
    all_similarities = []

    for category, items in embeddings_data.items():
        for item in items:
            similarity = cosine_similarity(user_embedding, item["embedding"])
            all_similarities.append({
                "category": category,
                "text": item["text"],
                "similarity": similarity,
                "row_index": item["row_index"]
            })

    # Sort by similarity (highest first)
    all_similarities.sort(key=lambda x: x["similarity"], reverse=True)

    # Return top N results
    return all_similarities[:top_n]


def main():
    # Example usage
    user_query = "I'm homeless in Atlanta and I'd like to get back on my feet. Where can I start?"
    results = find_closest_embeddings(user_query, "embeddings.json")

    print("\nTop 3 most relevant results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Category: {result['category']} (Similarity: {result['similarity']:.4f})")
        # Print a preview of the text (first 150 characters)
        preview = result['text'][:150] + "..." if len(result['text']) > 150 else result['text']
        print(f"   {preview}")

    # Ask if user wants to see full text of any result
    choice = input("\nEnter a number to see full text (or press Enter to exit): ")
    if choice and choice.isdigit() and 1 <= int(choice) <= len(results):
        idx = int(choice) - 1
        print(f"\nFull text from {results[idx]['category']}:\n")
        print(results[idx]['text'])


if __name__ == "__main__":
    main()