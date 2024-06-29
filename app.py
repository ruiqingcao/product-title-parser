import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html
from dash.dash_table import DataTable
from dash.dependencies import Output, Input, State
import plotly.express as px
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from gliner_spacy.pipeline import GlinerSpacy
import warnings
import os

warnings.filterwarnings("ignore", message="The sentencepiece tokenizer")

# Initialize Dash app with Bootstrap theme and Font Awesome
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, 'https://use.fontawesome.com/releases/v5.8.1/css/all.css'])

# Create server variable
server = app.server

# Reference absolute file path 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CATEGORIES_FILE = os.path.join(BASE_DIR, 'google_categories(v2).txt')

# Configuration for GLiNER integration
custom_spacy_config = {
    "gliner_model": "urchade/gliner_small-v2.1",
    "chunk_size": 250,
    "labels": ["person", "organization", "location", "event", "work_of_art", "product", "service", "date", "number", "price", "address", "phone_number", "misc"],
    "style": "ent",
    "threshold": 0.3
}

# Model variables
nlp = None
sentence_model = None

# Function to load models
def load_models():
    global nlp, sentence_model
    nlp = spacy.blank("en")
    nlp.add_pipe("gliner_spacy", config=custom_spacy_config)
    sentence_model = SentenceTransformer('all-roberta-large-v1')

# Load Google's content categories
with open(CATEGORIES_FILE, 'r') as f:
    google_categories = [line.strip() for line in f]

# Function to perform NER using GLiNER with spaCy
def perform_ner(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Function to extract entities using GLiNER with spaCy
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities if entities else ["No specific entities found"]

# Function to precompute category embeddings
def compute_category_embeddings():
    return sentence_model.encode(google_categories)

# Function to perform topic modeling using sentence transformers
def perform_topic_modeling_from_similarities(similarities):
    top_indices = similarities.argsort()[-3:][::-1]
    
    best_match = google_categories[top_indices[0]]
    second_best = google_categories[top_indices[1]]
    
    if similarities[top_indices[0]] > similarities[top_indices[1]] * 1.1:
        return best_match
    else:
        return f"{best_match} , {second_best}"

# Function to sort keywords by intent feature
def sort_by_keyword_feature(f):
    if type(f) != str:
        return "other"
    f = f.lower()

    informational_keywords = [
        "advice", "help", "how do i", "how does", "how to", "ideas", "information", "tools", "list", 
        "resources", "tips", "tutorial", "diy", "ways to", "what does", "what is", "what was", "where are", "where does", 
        "where can", "where is", "where was", "when is", "when are", "when was", "where to", "who is", "who said", "who wrote", 
        "who are", "why are", "who was", "why is", "examples", "explained", "meaning of", "definition", "benefits of", "uses of", 
        "overview", "summary", "report", "study",  "analysis", "research", "insight", "data", "facts", "details", "background", 
        "context", "news", "history", "documentation", "article", "paper", "blog", "forum", "discussion", "commentary", 
        "opinion", "perspective", "viewpoint", "guide", "difference between", "types of"
    ]

    navigational_keywords = [
        "facebook", "meta", "twitter", "site", "login", "account", "official website", "homepage", "portal", 
        "signin", "register", "signup", "dashboard", "profile", "settings", "control panel", "main page", 
        "user area", "admin", "control", "access", "entry", "webpage", "navigate", "home", "site map", 
        "directory", "find", "search", "lookup", "index", "online", "internet", "web", "browser", "navigate to", 
        "goto", "landing page", "url", "hyperlink", "link", "web address", "navigate", 
        "web navigation", "website address", "app", "download", "status", "join"
    ]

    local_keywords = [
        "closest", "close", "near me", "my area", "residential", "my zip", "my city", "nearby", "in town", 
        "around here", "local", "near", "vicinity", "local area", "nearest", "surrounding", "within miles", 
        "in my neighborhood", "district", "zone", "region", "near my location", "local services", "community", 
        "local shop", "in my vicinity", "local store", "suburb", "urban area", "within walking distance", 
        "around my place", "within my reach", "close by", "local office", "local branch", "near me now", 
        "in my locale", "within the city", "local market", "in my town", "local spot", "local point", 
        "local guide", "near my house", "local venue", "close to me", "within blocks", "local attractions", 
        "local events", "address"
    ]

    commercial_keywords = [
        "best", "affordable", "budget", "cheap", "expensive", "review", "top", "service", "cost", "average cost", 
        "calculator", "provider", "company", "vs", "companies", "professional", "specialist", "compare", 
        "comparison", "rating", "testimonials", "recommendation", "advisor", "consultant", "expert", "ranking", 
        "leader", "top-rated", "best-selling", "trending", "featured", "highlighted", "recommended", "popular", 
        "favorite", "preferred", "choice", "most reviewed", "highest rated", "highly recommended", "award-winning", 
        "five-star", "customer favorite", "top pick", "critically acclaimed", "editor's choice", "people's choice", 
        "top performer", "best value", "best overall", "best quality", "best price", "most trusted", "leading brand", 
        "popular choice", "most popular", "fees", "pros and cons"
    ]

    transactional_keywords = [
        "price", "quotes", "pricing", "purchase", "rates", "how much", "same day", "same-day", "buy", "order", 
        "discount", "deal", "offers", "sale", "checkout", "book", "reservation", "reserve", "bargain", "coupon", 
        "promo", "rebate", "clearance", "markdown", "buy one get one", "bogo", "special", "exclusive", "bundle", 
        "package", "subscription", "membership", "payment", "installment", "financing", "contract", "billing", 
        "invoice", "ticket", "admission", "entry", "enrollment", "register", "sign up", "pre-order", "e-commerce", 
        "shopping cart"
    ]

    if any(keyword in f for keyword in informational_keywords):
        return "informational"
    if any(keyword in f for keyword in navigational_keywords):
        return "navigational"
    if any(keyword in f for keyword in local_keywords):
        return "local"
    if any(keyword in f for keyword in commercial_keywords):
        return "commercial investigation"
    if any(keyword in f for keyword in transactional_keywords):
        return "transactional"

    return "other"

# Optimized batch processing of keywords
def batch_process_keywords(keywords, batch_size=32):
    processed_data = {'Keywords': [], 'Intent': [], 'NER Entities': [], 'Google Content Topics': []}
    
    # Precompute keyword embeddings once
    keyword_embeddings = sentence_model.encode(keywords, batch_size=batch_size, show_progress_bar=True)
    
    # Compute category embeddings
    category_embeddings = compute_category_embeddings()
    
    for i in range(0, len(keywords), batch_size):
        batch = keywords[i:i+batch_size]
        batch_embeddings = keyword_embeddings[i:i+batch_size]
        
        # Batch process intents
        intents = [sort_by_keyword_feature(kw) for kw in batch]
        
        # Batch process entities
        entities = [extract_entities(kw) for kw in batch]
        
        # Batch process topics
        similarities = cosine_similarity(batch_embeddings, category_embeddings)
        Google_Content_Topics = [perform_topic_modeling_from_similarities(sim) for sim in similarities]
        
        processed_data['Keywords'].extend(batch)
        processed_data['Intent'].extend(intents)
        
        # Convert entities to strings, handling both tuples and strings
        processed_entities = []
        for entity_list in entities:
            entity_strings = []
            for entity in entity_list:
                if isinstance(entity, tuple):
                    entity_strings.append(f"{entity[0]} ({entity[1]})")
                else:
                    entity_strings.append(str(entity))
            processed_entities.append(", ".join(entity_strings))
        
        processed_data['NER Entities'].extend(processed_entities)
        processed_data['Google Content Topics'].extend(Google_Content_Topics)
    
    return processed_data

# Main layout of the dashboard
app.layout = dbc.Container([
    dcc.Store(id='models-loaded', data=False),
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("About", href="#about")),
            dbc.NavItem(dbc.NavLink("Contact", href="#contact")),
        ],
        brand="KeyIntentNER-T",
        brand_href="https://jeredhiggins.com/keyintentnert",
        color="#151515",
        dark=True,
        brand_style={"background": "linear-gradient(to right, #ff7e5f, #feb47b)", "-webkit-background-clip": "text", "color": "transparent", "textShadow": "0 0 1px #ffffff, 0 0 3px #ff7e5f, 0 0 5px #ff7e5f"},
    ),
    
    dbc.Row(dbc.Col(html.H1('Keyword Intent, Named Entity Recognition (NER), & Google Topic Modeling Dashboard', className='text-center text-light mb-4 mt-4'))),

    dbc.Row([
        dbc.Col([
            dbc.Alert(
                "Models are loading. This may take a few minutes. Please wait...",
                id="loading-alert",
                color="info",
                is_open=True,
            ),
            dbc.Label('Enter keywords (one per line, maximum of 100):', className='text-light'),
            dcc.Textarea(id='keyword-input', value='', style={'width': '100%', 'height': 100}),
            dbc.Button('Submit', id='submit-button', color='primary', className='mb-3', disabled=True),
            dbc.Alert(id='alert', is_open=False, duration=4000, color='danger', className='my-2'),
            dbc.Alert(id='processing-alert', is_open=False, color='info', className='my-2'),
        ], width=6)
    ], justify='center'),
    
    # Loading component
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading",
                type="default",
                children=[
                    html.Div([
                        html.Div(id="loading-output")
                    ], className="my-4")
                ],
            ),
        ], width=12)
    ], justify='center', className="mb-4"),  # Added margin-bottom for separation
    dbc.Row(dbc.Col(dcc.Graph(id='bar-chart'), width=12)),

    dbc.Row([
        dbc.Col([
            dbc.Label('View all keyword data for each intent category:', className='text-light mt-4'),
            dcc.Dropdown(
                id='table-intent-dropdown',
                options=[],
                placeholder='Select an Intent',
                className='text-dark'
            ),
        ], width=6)
    ], justify='center'),

    dbc.Row(dbc.Col(
        html.Div(id='keywords-table', style={'width': '100%'}),
        width=12
    )),

    dbc.Row(dbc.Col(
        dbc.Button('Download CSV For All Keywords', id='download-button', color='success', className='my-5', disabled=True),
        width=12
    ), justify='center'),

    dcc.Download(id='download'),
    dcc.Store(id='processed-data'),

    # Explanation content
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.H3([html.I(className="fas fa-info-circle mr-2"), "About KeyIntentNER-T"], className="card-title text-warning"),
                        html.P("This tool provides valuable keyword insights for SEO and digital marketing professionals. Enter a list of keywords and get insights into Keyword Intent, NLP Entities extracted via NER (Named Entity Recognition), & Topics. I created KeyIntentNER-T as an example of how to use more modern NLP methods to gain insights into shorter text strings (keywords) and how this information may be understood by search engines using similar techniques.", className="card-text"), 
                    ])
                ], className="mb-4 shadow-sm"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H3([html.I(className="fas fa-pen mr-2"), "Notes on the data"], className="card-title text-success"),
                                dbc.ListGroup([
                                    dbc.ListGroupItem([html.I(className="fas fa-check mr-2"), "Keyword Intent is determined using a custom function that looks for the presence of specific terms and then classifies it into one of six predefined intent categories: 'informational', 'navigational', 'local', 'commercial investigation', 'transactional', or 'other'."]),
                                    dbc.ListGroupItem([html.I(className="fas fa-check mr-2"), "NLP Entities are determined using GLiNER, an advanced Named Entity Recognition (NER) model that is better at classifying shorter text strings. Additionally, Entitites are mapped to all Entity Types included in the Google Cloud Natural Language API."]),
                                    dbc.ListGroupItem([html.I(className="fas fa-check mr-2"), "Topics are determined by matching keywords to topics from Google's well-known Content and Product taxonomies."]),
                                    dbc.ListGroupItem([html.I(className="fas fa-check mr-2"), "Since this tool is doing a lot behind the scenes, keyword processing can take anywhere from 30 seconds up to ~2 minutes."]),
                                ], flush=True)
                            ])
                        ], className="mb-4 shadow-sm")
                    ], md=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H3([html.I(className="fas fa-chart-line mr-2"), "Benefits for SEO"], className="card-title text-info"),
                                dbc.ListGroup([
                                    dbc.ListGroupItem([html.I(className="fas fa-arrow-up mr-2"), "Improved content strategy by focusing your SEO efforts on creating more relevant/helpful content that addresses the search intent for keywords."]),
                                    dbc.ListGroupItem([html.I(className="fas fa-bullseye mr-2"), "Enhanced keyword targeting by matching keywords to Google's well-known categories, ensuring your content is aligned with popular search themes."]),
                                    dbc.ListGroupItem([html.I(className="fas fa-users mr-2"), "Better understanding of what kind of information a person is looking for."]),
                                    dbc.ListGroupItem([html.I(className="fas fa-robot mr-2"), "Better understanding of how keywords can be interpreted by search engines."]),
                                ], flush=True)
                            ])
                        ], className="mb-4 shadow-sm")
                    ], md=6),
                ]),
                dbc.Card([
                    dbc.CardBody([
                        html.H3([html.I(className="fas fa-quote-left mr-2"), "GLiNER Model Citation"], className="card-title text-light"),
                        html.P([
                            "GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer. ",
                            html.Br(),
                            "Authors: Urchade Zaratiana, Nadi Tomeh, Pierre Holat, Thierry Charnois.",
                            html.Br(),
                            "Year: 2023.",
                            html.Br(),
                            html.A([html.I(className="fas fa-external-link-alt mr-2"), "arXiv:2311.08526"], href="https://arxiv.org/abs/2311.08526", target="_blank", className="btn btn-outline-warning btn-sm mt-2")
                        ], className="card-text"),
                    ])
                ], className="mb-4 shadow-sm")
            ], id="about")
        ], width=12)
    ], className="mt-5"),

    # Contact section
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.H3([html.I(className="fas fa-envelope mr-2"), "Contact"], className="card-title text-info"),
                        html.P([
                            "For questions or if you are interested in building custom SEO dash apps, contact me at: ",
                            html.A("jrad.seo@gmail.com", href="mailto:jrad.seo@gmail.com", className="text-info")
                        ], className="card-text"),
                    ])
                ], className="mb-4 shadow-sm")
            ], id="contact")
        ], width=12)
    ], className="mt-4 mb-4"),

    # Hidden divs for smooth scrolling
    html.Div(id='dummy-input', style={'display': 'none'}),
    html.Div(id='dummy-output', style={'display': 'none'}),

], fluid=True)

# Callback to load models and update the loading alert
@app.callback(
    [Output('models-loaded', 'data'),
     Output('loading-alert', 'is_open'),
     Output('submit-button', 'disabled')],
    [Input('models-loaded', 'data')]
)
def load_models_callback(loaded):
    if not loaded:
        load_models()
        return True, False, False
    return loaded, False, False

# Callback for smooth scrolling
app.clientside_callback(
    """
    function(n_clicks) {
        const links = document.querySelectorAll('a[href^="#"]');
        links.forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                const targetId = this.getAttribute('href').substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {
                    targetElement.scrollIntoView({behavior: 'smooth'});
                }
            });
        });
        return '';
    }
    """,
    Output('dummy-output', 'children'),
    Input('dummy-input', 'children')
)

# All other callbacks
@app.callback(
    Output('alert', 'is_open'),
    Output('alert', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('keyword-input', 'value')]
)
def limit_keywords(n_clicks, keyword_input):
    if n_clicks is None:
        return False, ""
    
    keywords = keyword_input.split('\n')
    if len(keywords) > 100:
        return True, "Maximum limit of 100 keywords exceeded. Only the first 100 keywords will be processed."
    
    return False, ""

@app.callback(
    [Output('processed-data', 'data'),
     Output('loading-output', 'children'),
     Output('processing-alert', 'is_open'),
     Output('processing-alert', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State('keyword-input', 'value')]
)
def process_keywords(n_clicks, keyword_input):
    if n_clicks is None or not keyword_input:
        return None, '', False, ''

    keywords = [kw.strip() for kw in keyword_input.split('\n')[:100] if kw.strip()]
    processed_data = batch_process_keywords(keywords)

    return processed_data, '', True, "Keyword processing complete!"

@app.callback(
    Output('bar-chart', 'figure'),
    [Input('processed-data', 'data')]
)
def update_bar_chart(processed_data):
    if processed_data is None:
        return {
            'data': [],
            'layout': {
                'height': 0,  # Set height to 0 when there's no data
                'annotations': [{
                    'text': '',
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 28}
                }]
            }
        }

    df = pd.DataFrame(processed_data)
    intent_counts = df['Intent'].value_counts().reset_index()
    intent_counts.columns = ['Intent', 'Count']

    fig = px.bar(intent_counts, x='Intent', y='Count', color='Intent', 
                 title='Keyword Intent Distribution', 
                 color_discrete_sequence=px.colors.qualitative.Dark2)
    
    fig.update_layout(
        plot_bgcolor='#222222',
        paper_bgcolor='#222222',
        font_color='white',
        height=400,  # Set a fixed height for the chart
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

@app.callback(
    [Output('table-intent-dropdown', 'options'),
     Output('download-button', 'disabled')],
    [Input('processed-data', 'data')]
)
def update_dropdown_and_button(processed_data):
    if processed_data is None:
        return [], True

    df = pd.DataFrame(processed_data)
    intents = df['Intent'].unique()
    options = [{'label': intent, 'value': intent} for intent in intents]
    return options, False

@app.callback(
    Output('keywords-table', 'children'),
    [Input('table-intent-dropdown', 'value')],
    [State('processed-data', 'data')]
)
def update_keywords_table(selected_intent, processed_data):
    if processed_data is None or selected_intent is None:
        return html.Div()

    df = pd.DataFrame(processed_data)
    filtered_df = df[df['Intent'] == selected_intent]

    table = DataTable(
        columns=[{"name": i, "id": i} for i in filtered_df.columns],
        data=filtered_df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto', 'minWidth': '100px', 'width': '100px', 'maxWidth': '100px'},
        style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
        style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},
        sort_action='native',
        page_action='native',
        page_current=0
    )
    return table

@app.callback(
    Output('download', 'data'),
    [Input('download-button', 'n_clicks')],
    [State('processed-data', 'data')]
)
def download_csv(n_clicks, processed_data):
    if n_clicks is None or processed_data is None:
        return None

    df = pd.DataFrame(processed_data)
    csv_string = df.to_csv(index=False, encoding='utf-8')
    return dict(content=csv_string, filename="KeyIntentNER-T_keyword_analysis.csv")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if PORT is not set
    app.run_server(debug=True, host='0.0.0.0', port=port)
