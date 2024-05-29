import dash
import base64
import io
import nltk
import string
import re
import emoji
import pandas as pd
import plotly.express as px
from dash import dcc, html, dash_table
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentiment import sentiment_analyzer

# Text Cleaning
def clean_text(text):
    # Remove retweets
    text = re.sub(r'^RT @\w+: ', '', text)
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)

    # Remove URLs
    url_pattern = re.compile(r'((http|https):\/\/)?(www\.)?[\w\-\.]+(\.\w{2,})+(\/\S*)?')
    text = url_pattern.sub('', text)

    # Remove extra spaces, de-emojize, punctuation, and convert to lowercase
    text = emoji.demojize(re.sub('\s+', ' ', text).strip().translate(str.maketrans('', '', string.punctuation)).lower())

    # Tokenize, lemmatize, and remove stopwords
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    cleaned_text = ' '.join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])

    return cleaned_text

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the Dash app
app.layout = html.Div([
    # Header Section
    html.Div([
        html.Div([
            html.H1(
                "Brand Sentiment Analysis on Twitter",
                style={'fontSize': '24px', 'textAlign': 'left', 'marginBottom': '20px', 'marginLeft': '40px', 'fontFamily': 'Helvetica Neue, Helvetica'}
            ),
            html.Div([
                html.Div([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Button('Upload Data', style={'fontSize': '13px', 'fontFamily': 'Helvetica Neue, Helvetica'}),
                        style={'textAlign': 'left', 'marginBottom': '10px', 'position': 'sticky', 'top': '0'}
                    ),
                    dcc.Input(
                        id='keyword-input',
                        type='text',
                        placeholder='Enter keyword',
                        style={'fontSize': '12px', 'fontFamily': 'Helvetica Neue, Helvetica', 'textAlign': 'left', 'marginBottom': '10px', 'position': 'sticky', 'top': '0', 'width': '240px'}
                    ),
                    dcc.Store(id='dropdown-value'),
                    dcc.Dropdown(
                        id='text-col-dropdown',
                        placeholder='Select text column to be analyzed',
                        style={'fontSize': '12px', 'fontFamily': 'Helvetica Neue, Helvetica', 'width': '250px'}
                    )
                ], style={'textAlign': 'left', 'marginLeft': '40px'}),
                html.Div([
                    html.Div(id='output-data-upload', style={'textAlign': 'right', 'float': 'right', 'fontSize': '14px', 'fontFamily': 'Helvetica Neue, Helvetica'})
                ])
            ], style={'display': 'flex', 'justifyContent': 'space-between'})
        ], style={'position': 'fixed', 'top': '0', 'left': '0', 'zIndex': '1000', 'backgroundColor': '#f0f0f0', 'padding': '10px', 'width': '100%', 'boxSizing': 'border-box'}),

        html.Div([
            html.Button('Show Data', id='display-head-button', disabled=True, style={'marginRight': '10px', 'marginTop': '10px'}),
            html.Button('Process Data', id='process-data-button', disabled=True, style={'marginRight': '10px', 'marginTop': '10px'}),
            html.Button('Show Sentiment Distribution', id='sentiment-distribution-button', disabled=True, style={'marginRight': '10px', 'marginTop': '10px'}),
            html.Button('Show Sentiment Heatmap', id='sentiment-heatmap-button', disabled=True, style={'marginRight': '10px', 'marginTop': '10px'})
        ], style={'textAlign': 'center'})
    ], style={'width': '100%', 'marginTop': '185px'}),

    # Data Preview Section
    html.Div([
        html.H2("Data Preview", style={'textAlign': 'center', 'marginTop': '40px', 'fontFamily': 'Helvetica Neue, Helvetica'}),
        html.Div(id='display-head-output', style={'margin': '20px'})
    ], style={'width': '98%', 'margin': 'auto', 'marginTop': '40px'}),

    # Sentiment Analysis (Distribution and Heatmap) Sections
    html.Div([
        html.Div([
            html.H2("Sentiment Distribution", style={'textAlign': 'center', 'fontFamily': 'Helvetica Neue, Helvetica'}),
            html.Div(dcc.Graph(id='pie-chart', style={'visibility': 'hidden'}))
        ], style={'width': '45%', 'margin': 'auto', 'marginTop': '40px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'padding': '20px', 'borderRadius': '10px', 'boxSizing': 'border-box'}),
        html.Div([
            html.H2("Sentiment Heatmap", style={'textAlign': 'center', 'fontFamily': 'Helvetica Neue, Helvetica'}),
            html.Div(dcc.Graph(id='heatmap', style={'visibility': 'hidden'}))
        ], style={'width': '45%', 'margin': 'auto', 'marginTop': '40px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'padding': '20px', 'borderRadius': '10px', 'boxSizing': 'border-box'})
    ], style={'display': 'flex', 'justifyContent': 'space-between'})
])


# Define a session variable to store the uploaded data
app.config.suppress_callback_exceptions = True
session = {}

# Define callback to upload data
@app.callback(
    [Output('output-data-upload', 'children', allow_duplicate=True),
     Output('display-head-button', 'disabled'),
     Output('process-data-button', 'disabled'),
     Output('text-col-dropdown', 'options', allow_duplicate=True)],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
     State('upload-data', 'last_modified')],
    prevent_initial_call=True
)
def update_data(contents, filename, last_modified):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Data validation: Check if the file is a CSV file
        if filename.endswith('.csv'):
            # Create a Dataframe
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            df_en = df[df['Lang'] == 'en']

            # Store the uploaded data in the session
            session['uploaded_data'] = df_en.to_csv(index=False).encode()
            session['filename'] = filename

            # Display success message
            upload_success_message  =  html.Div([
                html.H5(f'{filename}', style={'fontSize': '14px'}),
                html.P(f'Updated At : {pd.Timestamp.now()}', style={'fontSize': '10px'})
            ])
            return upload_success_message, False, False, [{'label': col, 'value': col} for col in df.columns[1:].to_list()]
        else:
            # Display error message for invalid file format
            upload_error_message = html.Div([
                html.H5('Error: Invalid file format. Please upload a CSV file.', style={'fontSize': '14px'})
            ])
            return upload_error_message, True, True
    else:
        raise PreventUpdate

# Callback to store the selected dropdown value in dcc.Store
@app.callback(
    Output('dropdown-value', 'data'),
    [Input('text-col-dropdown', 'value')]
)
def store_dropdown_value(text_col):
    return text_col

# Define callback to display head of uploaded data
@app.callback(
    Output('display-head-output', 'children'),
    [Input('display-head-button', 'n_clicks')]
)
def display_head_data(n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    if n_clicks % 2 == 1:  # Toggle the display
        # Read the uploaded data into a DataFrame
        df = pd.read_csv(io.BytesIO(session['uploaded_data']))
        
        # Rename the first column to 'Index'
        df.columns.values[0] = ''
        
        # Create a Dash DataTable to display the head of the DataFrame
        table = dash_table.DataTable(
            id='head-table',
            columns=[{'name': col, 'id': col} for col in df.columns],
            data=df.head().to_dict('records'),
            style_table={'overflowX': 'auto'},  # Horizontal scroll
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_cell={'minWidth': '100px', 'width': '150px', 'maxWidth': '300px', 'whiteSpace': 'normal'},
        )

        return table

# Define callback to process data
@app.callback(
    [Output('output-data-upload', 'children', allow_duplicate=True),
     Output('text-col-dropdown', 'options', allow_duplicate=True)],
    [Input('process-data-button', 'n_clicks')],
    prevent_initial_call=True
)
def process_data(n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    # Read the uploaded data into a DataFrame
    df = pd.read_csv(io.BytesIO(session['uploaded_data']))
    
    # Example processing: Clean text data
    if 'text' in df.columns:
        df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Store the processed data back into the session
    session['uploaded_data'] = df.to_csv(index=False).encode()

    # Notification message
    filename = session['filename']
    upload_success_message  =  html.Div([
        html.H5(f'{filename}', style={'fontSize': '14px'}),
        html.P(f'Updated At : {pd.Timestamp.now()}', style={'fontSize': '10px'})
    ])
    return upload_success_message, [{'label': col, 'value': col} for col in df.columns[1:].to_list()]

# Define callback to enable/disable display pie chart button based on keyword input
@app.callback(
    [Output('sentiment-distribution-button', 'disabled'),
     Output('sentiment-heatmap-button', 'disabled')],
    [Input('keyword-input', 'value'),
     Input('upload-data', 'contents')]
)
def set_pie_chart_button_state(keyword, contents):
    if contents is None or not keyword:
        return True, True
    return False, False
    
# Define callback to display the pie chart
@app.callback(
    [Output('pie-chart', 'figure'),
     Output('pie-chart', 'style')],
    [Input('sentiment-distribution-button', 'n_clicks'),
     Input('dropdown-value', 'data')],
    [State('keyword-input', 'value')],
    prevent_initial_call=True
)
def display_pie_chart(n_clicks,selected_text_col,keyword):
    if n_clicks is None:
        raise PreventUpdate

    if keyword:
        # Read the uploaded data into a DataFrame
        df = pd.read_csv(io.BytesIO(session['uploaded_data']))

        text_col = selected_text_col or 'text'

        # Add column is_keyword
        df['is_keyword'] = df[text_col].str.contains(keyword, case=False, na=False).astype(int)

        # Do sentiment analysis
        sentiment_analyzer(df,text_col)

        # Group df by sentiment_label
        grouped_df = df[df['is_keyword'] == 1].groupby('sentiment_label').agg(
                    count=('sentiment_score', 'count'),
                    avg_value=('sentiment_score', 'mean')
                ).reset_index()
        
        # Define colors for each sentiment label
        colors = {
            'positive': '#00CC96',
            'neutral': '#636EFA',
            'negative': '#EF553B'
        }

        # Create a pie chart using Plotly Express
        fig = px.pie(
            grouped_df,
            names='sentiment_label',
            values='count',
            color='sentiment_label',
            color_discrete_map=colors
        )

        fig.update_traces(hoverinfo='label+value',
                          textinfo='label+percent'
        )
        return fig, {}
    
# Define callback to display the pie chart
@app.callback(
    [Output('heatmap', 'figure'),
     Output('heatmap', 'style')],
    [Input('sentiment-heatmap-button', 'n_clicks'),
     Input('dropdown-value', 'data')],
    [State('keyword-input', 'value')],
    prevent_initial_call=True
)
def display_heatmap(n_clicks,selected_text_col,keyword):
    if n_clicks is None:
        raise PreventUpdate

    if keyword:
        # Read the uploaded data into a DataFrame
        df = pd.read_csv(io.BytesIO(session['uploaded_data']))
        iso3 = pd.read_csv(r'./brand/iso3.csv',sep=';')

        text_col = selected_text_col or 'text'

        # Add column is_keyword
        df['is_keyword'] = df[text_col].str.contains(keyword, case=False, na=False).astype(int)

        # Do sentiment analysis
        sentiment_analyzer(df,text_col)

        df_merge = df.merge(iso3, left_on='Country', right_on='Country')

        # Group df by sentiment_label
        final = df_merge[df_merge['is_keyword'] == 1].groupby('iso3code').agg(
                    count=('sentiment_score', 'count'),
                    avg_value=('sentiment_score', 'mean')
                ).reset_index()
        
        # Rename the 'value' column to 'Value' for display purposes
        final.rename(columns={'count': 'Tweets Count',
                            'avg_value':'Avg Sentiment Score'}
                            ,inplace=True)

        # Calculate the color scale range
        vmin = final['Avg Sentiment Score'].min()
        vmax = final['Avg Sentiment Score'].max()
        vcenter = 0

        # Normalize the custom color scale for a continuous scale
        norm_color_scale = [
            (0, '#EF553B'),        # Red for the minimum value
            ((0 - vmin) / (vmax - vmin), 'white'),  # White for zero
            (1, '#00CC96')       # Green for the maximum value
        ]

        # Create the choropleth map
        fig = px.choropleth(final, 
                            locations='iso3code', 
                            locationmode='ISO-3',  # Use ISO-3 country codes
                            color='Avg Sentiment Score',
                            color_continuous_scale=norm_color_scale,
                            projection='natural earth',  # Choose the map projection
                            hover_name='iso3code'  # Tooltip shows country names
                            )

        # Update layout to set the zero value to white exactly
        fig.update_layout(coloraxis_colorbar=dict(
            title='Avg Sentiment Score',
            tickvals=[vmin, vcenter, vmax],
            ticktext=[str(vmin), '0', str(vmax)]
        ))

        return fig, {}

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)