from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI()


@app.get('/developer/')
def developer(developer):
    df = pd.read_csv('endpoint_1.csv')
    d = df[df.developer == developer]
    final_dev = pd.DataFrame(d.groupby(['developer', 'year'])['id'].count())
    final_dev['porcentaje'] = final_dev['id'] / 7793
    final_dev.reset_index(inplace=True)
    return final_dev.to_dict()

# 2


@app.get('/userdata/')
def userdata(user_id):
    df = pd.read_csv('endpoint_2.csv')
    fil_end2 = df[(df.user_id == user_id)]
    e = pd.DataFrame(
        fil_end2.groupby(['user_id', 'items_count'])['price'].sum())
    e.reset_index(inplace=True)
    return e.to_dict()



# 3


@app.get('/UserForGenre/')
def UserForGenre(genre):
    end2 = pd.read_csv('endpoint_3.csv')
    # Verifica si el genero ingresado existe en el DataFrame
    if genre not in end2.columns:
        return "Invalid genre"

    # Filtra el DataFrame por el genero ingresado
    genre_df2 = end2[end2[genre] == 1]

    # Ordera el DataFrame por playtime_forever
    sorted_genre_df2 = genre_df2.sort_values(by='playtime_forever',
                                             ascending=False)

    # Extrae los primeros 5 usuarios
    top_5 = sorted_genre_df2.head(5)

    # Crea un diccionario con los usuarios y sus datos
    top_users_dict = {}
    for _, row in top_5.iterrows():
        top_users_dict[row['user_id']] = {
            'year': row['year'],
            'playtime_forever': row['playtime_forever']
        }
    return top_users_dict


# 4


@app.get('/best_developer_year/')
def best_developer_year(year):
    df = pd.read_csv('endpoint_4.csv')
    fil_end4 = df[(df.year == year)]
    f = pd.DataFrame(
        fil_end4.groupby(['user_id',
                          'year'])['id'].count().sort_values(ascending=False))
    f.reset_index(inplace=True)
    s = f.head(3)
    return {
        'user_id': s.user_id.to_list(),
        'year': s.year.to_list(),
        'cantidad': s.id.to_list()
    }

#5


@app.get('/developer_reviews_analysis/')
def developer_reviews_analysis(developer):
    df = pd.read_csv('endpoint_5.csv')
    r = df[df.developer == developer]
    d = r.groupby(['developer'])[['user_id', 'positivo', 'negativo']].count()
    d.reset_index(inplace=True)
    return d.to_dict()


# ML
@app.get('/recomendacion/')
def recomendacion(title: str) -> list:
    df = pd.read_csv('reco.csv')

    # Combinamos reviews por titulos

    df["review"] = df["review"].fillna("")
    grouped = df.groupby('item_name').agg(lambda x: ' '.join(x)).reset_index()

    # 2. Calcula matriz TF-IDF usando stop words en inglés
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(grouped['review'])

    # Calcula matriz de similaridad del coseno
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    idx = grouped.index[grouped['item_name'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    item_indices = [i[0] for i in sim_scores]
    return grouped['item_name'].iloc[item_indices].tolist()
