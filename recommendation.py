from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
data = {
    'title': [
        'The Matrix', 'John Wick', 'The Lord of the Rings', 
        'The Hobbit', 'Harry Potter', 'Inception',
        'Interstellar', 'The Dark Knight', 'Fight Club', 'The Prestige'
    ],
    'description': [
        'A computer hacker learns about the true nature of reality and his role in the war against its controllers.',
        'An ex-hitman comes out of retirement to track down the gangsters that killed his dog.',
        'A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring.',
        'A reluctant Hobbit sets out to help a group of dwarves reclaim their mountain home from a dragon.',
        'A young wizard attends a school of witchcraft and wizardry as he discovers his magical heritage.',
        'A thief who steals corporate secrets through dream-sharing technology is given a final chance at redemption.',
        'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.',
        'Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.',
        'An office worker and a soap salesman build a global organization to help vent male aggression.',
        'Two stage magicians engage in a competitive rivalry that leads them on a life-long battle for supremacy.'
    ]
}


df = pd.DataFrame(data)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()
def recommend(title, num_recommendations=5):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]
print("Recommendations for 'The Matrix':")
print(recommend('The Matrix'))

