from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load CSV data
books = pd.read_csv('D:/BookRecommendation/model/Books.csv')
users = pd.read_csv('D:/BookRecommendation/model/Users.csv')
ratings = pd.read_csv('D:/BookRecommendation/model/Ratings.csv')

# Merge books and ratings for book info
ratings_with_bookName = ratings.merge(books, on='ISBN')

# Popularity dataframe (for homepage popular books)
num_rating_df = ratings_with_bookName.groupby('Book-Title').count()[['Book-Rating']].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num-ratings'}, inplace=True)

avg_rating_df = ratings_with_bookName.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg-rating'}, inplace=True)

popularity_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
popularity_df = popularity_df[popularity_df['num-ratings'] >= 250].sort_values('avg-rating', ascending=False).head(50)
popularity_df = popularity_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M', 'num-ratings', 'avg-rating']]

# Prepare pivot table for collaborative filtering
x = ratings_with_bookName.groupby('User-ID').count()['Book-Rating'] > 200
users_rating = x[x].index

filtered_rating = ratings_with_bookName[ratings_with_bookName['User-ID'].isin(users_rating)]

y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index

final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)

# Compute similarity matrix
similarity_scores = cosine_similarity(pt)

# Normalize function for search
def normalize(text):
    text = str(text)
    return ''.join(e.lower() for e in text if e.isalnum() or e.isspace()).strip()

# Add normalized columns to books DataFrame
books['normalized_title'] = books['Book-Title'].apply(normalize)
books['normalized_author'] = books['Book-Author'].apply(normalize)

@app.route('/', methods=['GET', 'POST'])
def index():
    data = []
    message = None

    if request.method == 'POST':
        user_input = request.form.get('user_input')
        normalized_input = normalize(user_input)

        # Exact author/title matches
        exact_author_match = books[books['normalized_author'] == normalized_input]
        exact_title_match = books[books['normalized_title'] == normalized_input]

        if not exact_author_match.empty:
            for _, row in exact_author_match.iterrows():
                data.append([
                    row['Book-Title'],
                    row['Book-Author'],
                    row['Image-URL-M']
                ])

        elif not exact_title_match.empty:
            book_row = exact_title_match.iloc[0]
            data.append([
                book_row['Book-Title'],
                book_row['Book-Author'],
                book_row['Image-URL-M']
            ])

            normalized_title = normalize(book_row['Book-Title'])
            pt_normalized_index = [normalize(title) for title in pt.index]

            if normalized_title in pt_normalized_index:
                pt_index = pt_normalized_index.index(normalized_title)
                similar_items = sorted(
                    list(enumerate(similarity_scores[pt_index])),
                    key=lambda x: x[1], reverse=True
                )[1:11]

                for i in similar_items:
                    similar_title = pt.index[i[0]]
                    temp = books[books['Book-Title'] == similar_title].drop_duplicates('Book-Title')
                    if not temp.empty:
                        data.append([
                            temp['Book-Title'].values[0],
                            temp['Book-Author'].values[0],
                            temp['Image-URL-M'].values[0]
                        ])
            else:
                message = "Book found, but not in the recommendation system."

        else:
            # Partial matches fallback
            partial_match = books[
                books['normalized_title'].str.contains(normalized_input, na=False) |
                books['normalized_author'].str.contains(normalized_input, na=False)
            ]

            if not partial_match.empty:
                for _, row in partial_match.iterrows():
                    data.append([
                        row['Book-Title'],
                        row['Book-Author'],
                        row['Image-URL-M']
                    ])
            else:
                message = "No book or author found. Showing popular books instead."

    return render_template(
        'index.html',
        book_name=list(popularity_df['Book-Title'].values),
        author=list(popularity_df['Book-Author'].values),
        image=list(popularity_df['Image-URL-M'].values),
        votes=list(popularity_df['num-ratings'].values),
        rating=list(popularity_df['avg-rating'].values),
        data=data,
        message=message
    )

@app.route('/book/<title>')
def book_details(title):
    normalized_title = normalize(title)
    match = books[books['normalized_title'] == normalized_title]

    if match.empty:
        return f"<h2>Book titled '{title}' not found.</h2>", 404

    book = match.iloc[0]
    return render_template('book_details.html', book=book)

if __name__ == '__main__':
    app.run(debug=True)
