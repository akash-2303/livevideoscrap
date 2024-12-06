import json
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])

def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.error(f"Error loading JSON file: {e}")
        return []

# Convert time string to datetime, using a reference date
def convert_time(time_str):
    reference_date = datetime(1900, 1, 1)
    try:
        dt = datetime.strptime(time_str, '%H:%M:%S')
    except ValueError:
        dt = datetime.strptime(time_str, '%M:%S')
    return reference_date + timedelta(hours=dt.hour, minutes=dt.minute, seconds=dt.second)

# Convert JSON data to DataFrame and process times
def prepare_data(data):
    segments = []
    for item in data:
        segment = item['segment']
        segment['linked_comments'] = item['linked_comments']
        segment['start'] = convert_time(segment['start'])
        segment['end'] = convert_time(segment['end'])
        segments.append(segment)
    df = pd.DataFrame(segments)
    logging.info(f"DataFrame Columns: {df.columns}")
    logging.info(f"DataFrame Head: \n{df.head()}")
    return df

# Create Gantt chart with correct datetime formatting
def create_gantt_chart(df):
    fig = px.timeline(
        df,
        x_start="start",
        x_end="end",
        y="speaker",
        custom_data=["message"],  # Adding message to custom data
        title="Debate Segments"
    )
    fig.update_traces(marker=dict(line=dict(color="black", width=1)))
    fig.update_yaxes(categoryorder="total ascending")
    
    # Update the x-axis to show only minutes and seconds
    fig.update_layout(
        xaxis_tickformat="%M:%S",
        xaxis=dict(
            dtick=60000  # dtick in milliseconds: 60000 ms = 1 minute
        ),
        xaxis_title='Time'
    )
    return fig

# Path to your JSON data file
json_file_path = 'output/D1.json'

# Load JSON data
data = load_json(json_file_path)
df_segments = prepare_data(data)

# Check if the DataFrame contains expected columns
expected_columns = ['speaker', 'start', 'end', 'message', 'linked_comments']
missing_columns = [col for col in expected_columns if col not in df_segments.columns]
if missing_columns:
    logging.error(f"Missing expected columns: {missing_columns}")
else:
    logging.info("All expected columns are present.")

# Create Gantt chart figure
fig = create_gantt_chart(df_segments)

# Create Dash app
app = dash.Dash(__name__)

@app.callback(
    Output('linked-comments', 'children'),
    [Input('gantt-chart', 'clickData')]
)
def display_linked_comments(clickData):
    if clickData is None:
        logging.info("No segment clicked yet.")
        return "Click on a segment to see linked comments."

    logging.info("Segment clicked.")
    logging.info(f"Click data: {clickData}")
    
    point = clickData['points'][0]
    segment_message = point['customdata'][0]  # Fetching the message from custom data
    
    logging.info(f"Segment Message: {segment_message}")
    
    # Find the corresponding segment
    segment = df_segments[df_segments['message'] == segment_message].iloc[0]
    linked_comments = segment['linked_comments'][:10] ## Displaying first 10 linked comments for debugging
    
    # Segment and linked comments HTML display
    display_contents = [
        html.H4("Segment Message:"),
        html.P(segment_message),
        html.Hr(),
        html.H4("Linked Comments:")
    ]
    
    for comment in linked_comments:
        display_contents.append(
            html.Div([
                html.H5(comment['author']),
                html.P(f"Published at: {comment['published_at']}"),
                html.P(f"Network: {comment['network']}"),
                html.Blockquote(comment['comment_text']),
                html.Hr()
            ])
        )
    
    return display_contents

# Layout of the app
app.layout = html.Div([
    dcc.Graph(id='gantt-chart', figure=fig),
    html.Div(id='linked-comments', children="Click on a segment to see linked comments.")
])

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)