import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"
THRESHOLD = 0.30  # 30% similarity threshold


def load_documents(folder):
    documents = {}
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                documents[file] = f.read()
    return documents


def compute_similarity(docs):
    filenames = list(docs.keys())
    texts = list(docs.values())

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    similarity_matrix = cosine_similarity(tfidf_matrix)
    return filenames, similarity_matrix


def detect_plagiarism(filenames, sim_matrix):
    print("\n🔍 Plagiarism Detection Results:\n")
    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            score = sim_matrix[i][j]
            percent = score * 100

            if score > THRESHOLD:
                print(f"⚠️  {filenames[i]} ↔ {filenames[j]}")
                print(f"    Similarity: {percent:.2f}%  ❌ Possible Plagiarism\n")
            else:
                print(f"✅  {filenames[i]} ↔ {filenames[j]}: {percent:.2f}%")


if __name__ == "__main__":
    docs = load_documents(DATA_DIR)
    files, similarity = compute_similarity(docs)
    detect_plagiarism(files, similarity)