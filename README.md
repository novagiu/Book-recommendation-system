# A Book Recommendation System Considering Contents and Emotions of User Interests

## Literature Review 

The article by Takumi Fujimoto and Harumi Murakami, titled “A Book Recommendation System
Considering Contents and Emotions of User Interests” was published in 2022 at the 12th
International Congress on Advanced Applied Informatics (IIAI-AAI)1. The authors propose a
novel book recommendation system that considers both the content and emotions of user
interests. The proposed method identifies recommended books based on the similarity of the
vectors of contents and emotions, contained in tweets about the content of user interests and
book reviews.

## Problem Description

The problem addressed in this report is the development of a book recommendation system
that takes into account both the content and emotions of user interests. Traditional
recommendation systems often rely solely on content-based or collaborative filtering
approaches. However, these methods might not fully capture the emotional context of user
preferences, especially in dynamic environments like social media.

## Proposed Solution

The proposed solution is a book recommendation system that integrates content and
sentiment analysis of user interests. Using user-generated content from Twitter and book
reviews from Amazon, the system identifies recommended books based on the similarity of
content vectors and emotional tones obtained from user tweets and book reviews.

---

## Methodology

- **Sentiment Analysis**: The VADER (Valence Aware Dictionary and sEntiment Reasoner) toolVADER (Valence Aware Dictionary and sEntiment Reasoner) tool.
- **Content Vectorization**: The model utilized for this task is the SentenceTransformer model 'all-
MiniLM-L6-v2'.
- **Similarity Calculation**: I calculated the cosine similarity between the user vectors and book vectors
to determine how similar they are.

---

## System Performance

The performance of the recommendation system can be assessed based on the
recommendations generated for a sample of users. Here are the results for three users:

![image](https://github.com/user-attachments/assets/89902c96-2d46-4112-944a-22ea9aed6f2b)


---

## How to Run

1. **Clone the repository**: Describes how to clone the repository to the local machine.
2. **Install dependencies**: Explains how to install all necessary dependencies using `pip` and `requirements.txt`.
3. **Run Jupyter Notebook**: Provides instructions on running the project in Jupyter Notebook.
4. **(Optional) Run in Google Colab**: Mentions the option to run the project in Google Colab for convenience.

---

## Author
Iuliia Ivanova  
Feel free to reach out via [https://www.linkedin.com/in/novagiu/]
