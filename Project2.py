import os
import numpy as np
import pandas as pd
import dash
from dash import dcc, html
import pandas as pd
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import imdb
import dash_bootstrap_components as dbc
import requests
from PIL import Image
from io import BytesIO
import cachetools
import functools
import base64

metadata_item_page = pd.read_csv('metadata_item_page.csv')
data_FM = pd.read_csv("data_FM.csv")
links_df = pd.read_csv('links.csv')
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')
ratings_df['userId'] = ratings_df['userId'].astype(int)
tags_df = pd.read_csv('tags.csv')
similarity_df = pd.read_csv('movies_similarity_df.csv', index_col=0)

dfs = [links_df, movies_df, ratings_df, tags_df]

# Cache 100 entries for 1 hour
cache = cachetools.TTLCache(maxsize=100, ttl=60 * 60)

# Loop through each dataframe
for df in dfs:
    # Check if 'timestamp' column exists
    if 'timestamp' in df.columns:
        # Convert 'timestamp' column to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

ratings_tags_df = pd.merge(ratings_df.merge(
    movies_df, on='movieId'), tags_df, on=['userId', 'movieId'])
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css', '/assets/st.css', dbc.themes.BOOTSTRAP]


def pad_with_zeros(string):
    return string.zfill(7)


def fetch_movie_image(imdb_id):
    # IMDb API URL to fetch movie details
    imdb_id = pad_with_zeros(imdb_id)
    imdb_api_url = f"http://www.omdbapi.com/?i=tt{imdb_id}&apikey=5661e40c"
    print(imdb_api_url)

    try:
        # Sending a GET request to the IMDb API
        response = requests.get(imdb_api_url)
        data = response.json()

        # Checking if the response contains an image URL
        if 'Poster' in data and data['Poster'] != 'N/A':
            # Fetching the image from the URL
            image_response = requests.get(data['Poster'])
            image_data = image_response.content

            # Returning the image data
            return BytesIO(image_data)
        else:
            print("No poster available for this movie.")
            return None
    except Exception as e:
        print(f"Error fetching movie image: {str(e)}")
        return None


app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)

app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(
            html.H1(
                'Movies Recommender System App',
                style={
                    'font-family': 'Century Gothic',
                    'font-weight': 'bold',
                    'color': 'white',
                    'text-align': 'center',  # Center align the text
                    'margin-top': '5px',  # Add some top margin for spacing
                    'margin-bottom': '5px',  # Add some bottom margin for spacing
                }
            ),
            width={'size': 6, 'offset': 3}  # Center the column horizontally
        ),
        style={
            'background-color': 'rgba(0, 0, 0, 0.3)',  # Dark background color
            'padding': '5px',  # Add padding for spacing
            'border-radius': '10px',  # Rounded corners for aesthetics
            # Add shadow for depth
            'box-shadow': '0px 4px 6px rgba(0, 0, 0, 0.1)',
            'margin-bottom': '5px',  # Add margin at the bottom for spacing
        }
    ),
    dbc.Tabs(id='tabs', active_tab='user-page', children=[
        dbc.Tab(label='User Page', tab_id='user-page'),
        dbc.Tab(label='Explore Movies', tab_id='item-page'),
        dbc.Tab(label='User Recommendations [DFM]',
                tab_id='user-recommendations'),
        dbc.Tab(label='User Recommendations [FM]', tab_id='item-similarity')
    ]),
    html.Div(id='page-content')
], fluid=True,
    style={'background-image': 'url(/assets/bk.PNG)',  # Set background image URL
           'background-size': 'cover',  # Cover the entire container
           'background-position': 'center',  # Center the background image
           'padding': '20px', 'opacity': '1', 'height': '750px'}
)


@app.callback(Output('page-content', 'children'),
              [Input('tabs', 'active_tab')])
def render_content(tab):
    if tab == 'user-page':
        return dbc.Container([
            dbc.Row([
                dbc.Col(html.Div([
                    html.Label('Select User ID:', style={
                               'font-weight': 'bold'}),
                    dcc.Dropdown(
                        id='user-dropdown',
                        placeholder="Select a user!",
                        options=[{'label': str(user_id), 'value': user_id}
                                 for user_id in ratings_df['userId'].unique()],
                        value=ratings_df['userId'].iloc[0],
                    ),
                ]), width=12)
            ]),
            html.Div(id='user-history')
        ], fluid=True)
    elif tab == 'item-page':
        return dbc.Container([
            dbc.Row([
                dbc.Col(dcc.Dropdown(
                    id='movie-dropdown',
                    placeholder="Select a movie",
                    options=[{'label': movie_title, 'value': movieId} for movie_title, movieId in zip(
                        metadata_item_page['movie_title'], metadata_item_page['movieId'])],
                    value=movies_df['movieId'].iloc[0],
                ), width=5)
            ]),
            html.Div(id='metadata'),
            html.Div(id='ratings_tags', style={'top' : '-50px'})
        ], fluid=True)
    elif tab == 'user-recommendations':
        return dbc.Container([
            dbc.Row([
                dbc.Col(html.Div([
                    html.Label('Select User ID:', style={
                               'font-weight': 'bold'}),
                    dcc.Dropdown(
                        id='user-dropdown-recommendations',
                        placeholder="Select a user!",
                        options=[{'label': str(user_id), 'value': user_id}
                                 for user_id in ratings_df['userId'].unique()],
                        value=ratings_df['userId'].iloc[0],
                    ),
                ]), width=12)
            ]),
            html.Div(id='user-recommendations-content')
        ], fluid=True)
    elif tab == 'item-similarity':
        return dbc.Container([
            dbc.Row([
                dbc.Col(html.Label('Select a user!'), width=2),
                dbc.Col(dcc.Dropdown(
                    id='user-dropdown-recommendations',
                    placeholder="Select a user!",
                    options=[{'label': str(user_id), 'value': user_id}
                             for user_id in ratings_df['userId'].unique()],
                    value=ratings_df['userId'].iloc[0],
                ), width=10)
            ]),
            html.Div(id='item-similarity-content')
        ], fluid=True)


@app.callback(Output('user-history', 'children'),
              [Input('user-dropdown', 'value')])
def update_user_history(selected_user):
    user_history_df = ratings_df[ratings_df['userId']
                                 == selected_user]
    user_history_df = pd.merge(user_history_df, movies_df, on=[
                               'movieId'], how='left')
    user_history_df = pd.merge(user_history_df, links_df, on=[
                               'movieId'], how='left')
    return generate_table(user_history_df, 'u')


@app.callback(
    [Output('metadata', 'children'),
     Output('ratings_tags', 'children')],
    [Input('movie-dropdown', 'value')]
)
def update_movie_history(selected_movie):

     movie_info = metadata_item_page[metadata_item_page['movieId'] == selected_movie]


        card_body = dbc.CardBody([
            update_movie_image(
                str(movie_info['imdbId'].unique()[0])),
            html.H4(movie_info['movie_title'],style={'font-size': '15px', 'font-weight': 'bold'}, className="card-title"),
            dbc.Badge(movie_info['genres'], pill=True, color="#810811", className="genre-badge", style={'font-size': '13px'}),
          html.Div([
            html.Img(src="https://th.bing.com/th/id/R.52cbca1d253bb459d0e5d37a0474e0b5?rik=CxNIDqkKYj4v8w&riu=http%3a%2f%2fwww.clipartbest.com%2fcliparts%2fdT8%2fMrR%2fdT8MrRKTe.png&ehk=9OTFOytpUajo6T0IHRxjljY1AiX1Q%2fs%2f2BGA%2bkHGLBk%3d&risl=&pid=ImgRaw&r=0", 
                     className="img-fluid",  # Make the image responsive
                     style={'max-width': '35px', 'margin-right': '10px'}  # Adjust the maximum width of the image and float it to the left
            ),
            html.H4(f"{float(movie_info.iloc[0, 4])}/5", style={'font-size': '17px', 'font-weight': 'bold', 'display': 'inline-block'}, className="card-title")
        ], style={'margin-top': '10px'}),  # Add margin top to create space between the elements
        html.H4(f"{int(movie_info.iloc[0, 5])} Rating(s)",style={'font-size': '13px','color':"#4B4341", 'margin-left': '40px', 'margin-top': '-10px'}, className="num_ratingg"),

        ])

        movie_card = dbc.Card(card_body, style={
            "padding": "10px",
            "border": "1px solid #ddd",
            "text-align": "center",
            "margin": "10px",
            "width": "calc(100% - 100px)"})

        # Create table for ratings and tags

        ratings_tags_df = pd.merge(ratings_df.drop(columns='timestamp'), tags_df.drop(columns='timestamp'), on=['userId', 'movieId'], how='left')
        ratings_tags_df['tag'] = ratings_tags_df['tag'].fillna('No tags yet')


        movie_ratings_tags_to_view = ratings_tags_df[ratings_tags_df['movieId'] == selected_movie]
        movie_ratings_tags_to_view.drop(columns='movieId',inplace=True)
        movie_ratings_tags_table = generate_table_m(movie_ratings_tags_to_view)

        recommended_movies = get_movie_recommendations(selected_movie)
        mask = movies_df['title'].isin(recommended_movies)
        movies_imdb = pd.merge(movies_df, links_df, on='movieId', how='left')

        # Apply the mask to filter rows and select desired columns
        similar_movies = movies_imdb.loc[mask]



        return  dbc.Row([
            dbc.Col([movie_card], width=4),  # Adjust width based on your design
            dbc.Col([html.P("Ratings and Tags:", style={
                'font-size': '16px',
                'color': 'black',
                'font-weight': 'bold',
                'top':'100px'
            }),dcc.Graph(figure=movie_ratings_tags_table)], width=4)  # Adjust width based on your design
        ]), dbc.Col([html.P('Similar Movies:', style={
                'font-size': '16px',
                'color': 'black',
                'font-weight': 'bold',
                'top':'100px','background-color': '#FCFCFC',  # Background color
                'padding': '5px',  # Add padding for better visibility
                'border-radius': '5px'
            }), generate_table(similar_movies, 'u')], width=15)



userId = 0
DMFadata = pd.read_csv('allpredictionsdata.csv')
alldata = pd.read_csv('dataformapping.csv')
# FMdata=pd.read_csv('')
# saraalldata[['movieId'].isin(saraalldata[userId].values)]


@app.callback(Output('user-recommendations-content', 'children'),
              [Input('user-dropdown-recommendations', 'value')])
def update_user_recommendations(selected_user):
    datatoshow = alldata[alldata['movieId'].isin(DMFadata[str(selected_user)])].drop(
        columns=['movieId'], axis=1)
    return dbc.Card(generate_table(datatoshow, "u", cards_per_row=5), style={'padding': '20px', 'margin': '20px'})


@app.callback(Output('item-similarity-content', 'children'),
              [Input('user-dropdown-recommendations', 'value')])
def update_item_similarity(selected_user):
    user_recommendations = data_FM[data_FM['userId'] == selected_user]
    unique_movie_ids = user_recommendations['movieId'].unique()
    recommended_movies_df = movies_df[movies_df['movieId'].isin(
        unique_movie_ids)]
    datatoshow = pd.merge(recommended_movies_df, links_df,
                          on='movieId', how='left')
    return dbc.Card(generate_table(datatoshow, "u", cards_per_row=5), style={'padding': '20px', 'margin': '20px'})


def generate_table_m(dataframe, max_rows=7):
    # Table header
    header_names = ['User ID', 'Rating', 'Tag']  # Replace with actual header names

    trace = go.Table(
        header=dict(values=header_names,
                    fill=dict(color='#FCFDFE'),
                    align=['left'], height=30 ),  # Adjust alignment to match the number of columns
        cells=dict(values=[dataframe.userId, dataframe.rating, dataframe.tag],
                   fill=dict(color='#F5F2FF'),
                   align=['left'] * 3, 
                  height=30),
        columnwidth=[30, 30, 30]
    )  # Adjust alignment to match the number of columns

    # Create a plotly figure with the table trace
    figure = go.Figure(data=[trace])
    figure.update_layout(margin=dict(l=0, r=0, t=0, b=0))


    # Display the plotly figure
    return figure

def get_movie_recommendations(movieId, num_recommendations=10):
    
    movie_title = movies_df[movies_df['movieId']==movieId].iloc[0,1]
    # Get the similarity values for the given song
    movie_similarities = similarity_df[movie_title].sort_values(ascending=False)
    
    # Get the most similar songs (excluding the input song itself)
    recommended_movies = movie_similarities.iloc[1:num_recommendations+1].index.tolist()
    
    return recommended_movies


def update_movie_image(imdb_id):
    """Displays a placeholder or fetches the movie image using lazy loading."""
    if imdb_id:
        image_data = fetch_movie_image(imdb_id)
        if image_data:
            # Convert BytesIO object to base64-encoded string
            encoded_image = base64.b64encode(
                image_data.getvalue()).decode('utf-8')
            return dbc.CardImg(src=f"data:image/png;base64,{encoded_image}", style={'width': '70%'})
        else:
            # Placeholder image
            return dbc.CardImg(src="https://cdn-icons-png.freepik.com/256/6287/6287678.png", style={'width': '70%'})
    else:
        # Placeholder image
        return dbc.CardImg(src="https://cdn-icons-png.freepik.com/256/6287/6287678.png", style={'width': '70%'})


def generate_table(dataframe, c, cards_per_row=5):

    # Prepare movie or user details based on card type
    if c == "u":
        details = dataframe[["title", "genres", "imdbId"]][:10]
    elif c == "m":
        details = dataframe[["userId", "rating"]]

    unique_cards = set()  # Set to store unique card indices
    cards = []

    for idx, entry in details.iterrows():
        if entry[0] not in unique_cards:
            unique_cards.add(entry[0])  # Add index to set for future checks
            # Create card body based on card type
            card_body = None
            if c == "u":
                genres_list = entry['genres'].split('|')
                formatted_genres = [dbc.Badge(genre, pill=True, color="#810811",
                                              className="genre-badge") for genre in genres_list]
                print(entry['title'], entry['imdbId'])
                card_body = dbc.CardBody([
                    update_movie_image(str(entry['imdbId'])),
                    html.H4(entry['title'], className="card-title"),
                    html.Div(formatted_genres, className="genres-container"),
                ])
            elif c == "m":
                card_body = dbc.CardBody([
                    html.H5(f"User ID: {entry['userId']}",
                            className="card-title"),
                    html.P(f"Rating: {entry['rating']}",
                           className="card-text"),

                ])

            if card_body:
                card = dbc.Card(card_body, style={
                    "padding": "10px",
                    "border": "1px solid #ddd",
                    "text-align": "center",
                    "margin": "10px",
                    "width": "calc(20% - 20px)"
                })
        cards.append(card)

    # Create a dbc row with 5 cards per row for the current page
    rows = [dbc.Row(cards[i:i+cards_per_row])
            for i in range(0, len(cards), cards_per_row)]

    # Wrap rows and pagination in a div for scrollability
    scrollable_div = html.Div([
        html.Div(rows, id='cards-container'),
    ], style={'overflowY': 'scroll', 'height': '400px'})

    return scrollable_div


# Running the app
if __name__ == '__main__':
    app.run_server(debug=True)
