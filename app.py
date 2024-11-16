from flask import Flask, render_template,request,jsonify
import pandas as pd
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
import plotly.graph_objects as go
import plotly.express as px
from bokeh.palettes import Viridis256
from bokeh.transform import dodge
from plotly.io import to_html
import plotly.io as pio
from plotly.subplots import make_subplots

app = Flask(__name__)

# Load and preprocess dataset
def load_data():
    dataset = pd.read_csv("cleaned_weatherAUS.csv")
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset['Year'] = dataset['Date'].dt.year
    dataset['Month'] = dataset['Date'].dt.month
    dataset = dataset.drop(columns=['RainTomorrow'])
    return dataset

dataset = load_data()
# Extract unique locations
def get_locations():
    return dataset['Location'].unique()


def create_weather_graph(dataset):
    dataset = pd.read_csv("cleaned_weatherAUS.csv")
    # Calculate 95th percentile thresholds
    temp_threshold = dataset['MaxTemp'].quantile(0.95)
    rain_threshold = dataset['Rainfall'].quantile(0.95)

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribution of Max Temperatures', 'Distribution of Rainfall'),
        shared_yaxes=True
    )

    # Add histogram for MaxTemp
    fig.add_trace(
        go.Histogram(x=dataset['MaxTemp'], nbinsx=30, marker_color='#0FA4AF', name='MaxTemp'),
        row=1, col=1
    )

    # Add 95th percentile line for MaxTemp
    fig.add_vline(x=temp_threshold, line=dict(color='red', dash='dash'), annotation_text=f'95th Percentile: {temp_threshold:.2f} °C', annotation_position="top right", row=1, col=1)

    # Add histogram for Rainfall
    fig.add_trace(
        go.Histogram(x=dataset['Rainfall'], nbinsx=30, marker_color='#0FA4AF', name='Rainfall'),
        row=1, col=2
    )

    # Add 95th percentile line for Rainfall
    fig.add_vline(x=rain_threshold, line=dict(color='red', dash='dash'), annotation_text=f'95th Percentile: {rain_threshold:.2f} mm', annotation_position="top right", row=1, col=2)

    # Update layout
    fig.update_layout(
        xaxis_title='Max Temperature (°C)',
        xaxis2_title='Rainfall (mm)',
        yaxis_title='Frequency',
        width=1200,
        showlegend=False
    )

    return fig


# Function to create the Plotly graph for extreme events
def create_extreme_events_graph(dataset):
    dataset = pd.read_csv("cleaned_weatherAUS.csv")
    # Define extreme event thresholds using the 95th percentile
    temp_threshold = dataset['MaxTemp'].quantile(0.95)
    rain_threshold = dataset['Rainfall'].quantile(0.95)

    # Filter data for the last 10 years
    last_10_years = dataset[dataset['Year'] >= (dataset['Year'].max() - 10)]

    # Calculate extreme events per year
    last_10_years['ExtremeTempEvent'] = last_10_years['MaxTemp'] > temp_threshold
    last_10_years['HeavyRainfallEvent'] = last_10_years['Rainfall'] > rain_threshold

    # Group by year and count occurrences of extreme events
    extreme_event_counts = last_10_years.groupby('Year').agg(
        ExtremeTempEvents=('ExtremeTempEvent', 'sum'),
        HeavyRainfallEvents=('HeavyRainfallEvent', 'sum')
    ).reset_index()

    # Create Plotly line plot
    fig = go.Figure()

    # Add extreme temperature events trace
    fig.add_trace(go.Scatter(
        x=extreme_event_counts['Year'],
        y=extreme_event_counts['ExtremeTempEvents'],
        mode='lines+markers',
        name='Extreme Temp Events',
        line=dict(color='#0FA4AF'),
        marker=dict(symbol='circle')
    ))

    # Add heavy rainfall events trace
    fig.add_trace(go.Scatter(
        x=extreme_event_counts['Year'],
        y=extreme_event_counts['HeavyRainfallEvents'],
        mode='lines+markers',
        name='Heavy Rainfall Events',
        line=dict(color='blue'),
        marker=dict(symbol='circle')
    ))

    # Update layout with extended axis ranges
    fig.update_layout(
        title='Frequency of Extreme Weather Events in Australia (Last 10 Years)',
        xaxis_title='Year',
        yaxis_title='Number of Extreme Events',
        paper_bgcolor='white',  # Set paper background color to white
        plot_bgcolor='white',   # Set plot background color to white
        showlegend=True,
        xaxis=dict(
            range=[extreme_event_counts['Year'].min() - 1, extreme_event_counts['Year'].max() + 1]  # Extend x-axis by 1 year on both sides
        ),
        yaxis=dict(
            range=[0, extreme_event_counts[['ExtremeTempEvents', 'HeavyRainfallEvents']].max().max() + 8]  # Extend y-axis based on maximum values
        )
    )

    return fig

# Plotly Scatter Plot Function
def create_plotly_scatter(location):
    dataset = pd.read_csv("cleaned_weatherAUS.csv")
    # Filter dataset for the specified location
    location_data = dataset[dataset['Location'] == location]
    
    # Create figure
    fig = go.Figure()

    # 9am Humidity vs Rainfall
    fig.add_trace(go.Scatter(
        x=location_data['Humidity9am'],
        y=location_data['Rainfall'],
        mode='markers',
        marker=dict(color='blue', opacity=0.5),
        name='9am Humidity'
    ))

    # 3pm Humidity vs Rainfall
    fig.add_trace(go.Scatter(
        x=location_data['Humidity3pm'],
        y=location_data['Rainfall'],
        mode='markers',
        marker=dict(color='#0FA4AF', opacity=0.5),
        name='3pm Humidity'
    ))

    # Update layout
    fig.update_layout(
        title=f'Relationship between Humidity and Rainfall in {location}',
        xaxis_title='Humidity (%)',
        yaxis_title='Rainfall (mm)',
        template='plotly',
        legend=dict(title="Humidity Time", x=1, y=0.9)
    )

    return fig.to_html(full_html=False)

# Bokeh Plot for Annual Rainfall by Region and Year
def create_bokeh_plott():
    dataset = pd.read_csv("cleaned_weatherAUS.csv")
    # Group and prepare data
    annual_rainfall = dataset.groupby(['Year', 'Location']).Rainfall.mean().reset_index()
    annual_rainfall['Location'] = annual_rainfall['Location'].astype(str)
    annual_rainfall['Year'] = annual_rainfall['Year'].astype(str)
    
    # Define unique years and locations
    years = list(annual_rainfall['Year'].unique())
    locations = annual_rainfall['Location'].unique()
    
    # Set up the figure
    p = figure(x_range=years, 
               height=600, width=1000,
               title="Average Annual Rainfall by Region (Grouped by Year)",
               toolbar_location=None, tools="")
    
    # Use the specified color for all bars
    custom_color = "#0FA4AF"

    # Plot each location's data
    num_locations = len(locations)
    for i, location in enumerate(locations):
        subset = annual_rainfall[annual_rainfall['Location'] == location]
        p.vbar(x=dodge('Year', -0.4 + i * (0.8 / num_locations), range=p.x_range),
               top='Rainfall', width=0.8 / num_locations, source=ColumnDataSource(subset),
               color=custom_color, legend_label=location)
    
    # Hover tool
    hover = HoverTool()
    hover.tooltips = [("Region", "@Location"), ("Average Rainfall (mm)", "@Rainfall"), ("Year", "@Year")]
    p.add_tools(hover)

    # Axis labels
    p.xaxis.axis_label = 'Year'
    p.yaxis.axis_label = 'Average Rainfall (mm)'
    p.xaxis.major_label_orientation = "vertical"

    # Legend settings
    p.legend.title = "Region"
    p.legend.click_policy = "mute"
    p.add_layout(p.legend[0], 'right')  # Move legend outside plot area

    return components(p)


def create_correlation_plot():
    dataset = pd.read_csv("cleaned_weatherAUS.csv")
    # Calculate correlation between MaxTemp and Rainfall for each city
    correlations = dataset.groupby('Location').apply(
        lambda group: group['MaxTemp'].corr(group['Rainfall'])
    ).reset_index()

    # Rename columns for clarity
    correlations.columns = ['Location', 'Correlation']

    # Sort correlations for better visualization
    correlations = correlations.sort_values(by='Correlation', ascending=False)

    # Create an interactive bar plot with Plotly
    fig = px.bar(
        correlations,
        x='Correlation',
        y='Location',
        orientation='h',
        color='Correlation',
       color_continuous_scale = [
                [0.0, '#003366'],  # Dark blue
                [0.25, '#336699'], # Dark-medium blue
                [0.5, '#0FA4AF'],  # Teal (specified color)
                [0.75, '#99BBDD'], # Light-medium blue
                [1.0, '#99CCFF']   # Light blue
            ], # Cu
        title='Correlation between Maximum Temperature and Rainfall by City',
        labels={'Correlation': 'Correlation Coefficient', 'Location': 'City'},
    )

    # Customize hover information to show the city and exact correlation value
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Correlation: %{x:.2f}'
    )

    # Update layout for readability
    fig.update_layout(
        xaxis_title="Correlation Coefficient",
        yaxis_title="City",
        yaxis=dict(autorange="reversed"),  # Keeps cities in sorted order from top to bottom
    )

    # Return the plot as an HTML div
    return to_html(fig, full_html=False)




# Plotly Heatmap (Humidity vs Rainfall)
def create_plotly_heatmap():
    dataset = pd.read_csv("cleaned_weatherAUS.csv")
    
    # Create a sample correlation dataframe 
    humidity_rainfall_corr = dataset.groupby('Location').apply(
        lambda x: x[['Humidity9am', 'Humidity3pm', 'Rainfall']].corr().loc['Rainfall', ['Humidity9am', 'Humidity3pm']]
    ).reset_index()

    # Set up the heatmap matrix
    correlation_matrix = humidity_rainfall_corr.set_index('Location')[['Humidity9am', 'Humidity3pm']].T

    # Define a blue-themed colorscale
    blue_colorscale = [
        [0.0, '#024950'],  # Dark blue
        [0.5, '#0FA4AF'],  # Teal (specified color)
        [1.0, '#99CCFF']   # Light blue
    ]
    # Create a Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,  # The correlation values
        x=correlation_matrix.columns,  # Locations (x-axis)
        y=correlation_matrix.index,  # Humidity Time (9am, 3pm)
        colorscale=blue_colorscale,  # Custom blue colorscale
        colorbar=dict(title='Correlation'),  # Colorbar with label
        text=correlation_matrix.values,  # Display the correlation values as text
        hoverinfo='text',  # Display text on hover
    ))

    # Update layout
    fig.update_layout(
        title='Correlation between Humidity and Rainfall by Location',
        xaxis_title='Rainfall of Location',
        yaxis_title='Humidity Time (9am and 3pm)',
        xaxis=dict(tickangle=-45),  # Rotate x-axis labels for better visibility
        template='plotly',  # Use the plotly default theme for styling
    )

    # Convert the plot to HTML for embedding in the template
    return fig.to_html(full_html=False)


def create_rainfall_plot(year):
    dataset = pd.read_csv("cleaned_weatherAUS.csv")
    # Group and filter data
    annual_rainfall = dataset.groupby(['Year', 'Location'])['Rainfall'].mean().reset_index()
    annual_rainfall['Year'] = annual_rainfall['Year'].astype(str)
    
    # Filter by the selected year
    rainfall_year = annual_rainfall[annual_rainfall['Year'] == year]
    locations = list(rainfall_year['Location'].unique())
    
    # Create data source
    source_year = ColumnDataSource(rainfall_year)
    
    # Set up the figure
    p = figure(x_range=locations, height=600, width=1000,
               title=f"Average Annual Rainfall by Region for Year {year}",
               toolbar_location=None, tools="")
    
    # Use the specified color for all bars
    custom_color = "#0FA4AF"  # Blue color
    
    # Add bars to the plot
    p.vbar(x='Location', top='Rainfall', width=0.8, source=source_year, color=custom_color)
    
    # Add hover tool
    hover = HoverTool(tooltips=[("Region", "@Location"), ("Average Rainfall (mm)", "@Rainfall")])
    p.add_tools(hover)
    
    # Customize plot
    p.xaxis.axis_label = 'Region'
    p.yaxis.axis_label = 'Average Rainfall (mm)'
    p.xaxis.major_label_orientation = "vertical"
    
    # Return script and div components for embedding
    script, div = components(p)
    return script, div



# Function to create the correlation heatmap
def create_correlation_heatmap(dataset_path):
    # Load dataset
    dataset = pd.read_csv("cleaned_weatherAUS.csv")
    # Select only numerical columns for the correlation matrix
    numerical_columns = dataset.select_dtypes(include=['float64', 'int64'])

    # Compute the correlation matrix
    corr_matrix = numerical_columns.corr()

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=[
                [0.0, '#024950'],  # Dark blue
                [0.5, '#0FA4AF'],  # Teal (specified color)
                [1.0, '#99CCFF']   # Light blue
            ],
        zmin=-1, zmax=1,
        hoverongaps=False,
        colorbar=dict(title='Correlation Coefficient'),
    ))

    # Customize layout
    fig.update_layout(
        title='Correlation Matrix of Numerical Features',
        xaxis_title='Features',
        yaxis_title='Features',
        autosize=True
    )

    return fig


@app.route('/', methods=['GET', 'POST'])
def index():
    # Default location
    selected_location = 'Sydney'

    if request.method == 'POST':
        selected_location = request.form.get('location', 'Sydney')

    # Only regenerate the Plotly scatter plot for the selected location
    plotly_scatter = create_plotly_scatter(selected_location)

    # Get the list of available locations for the dropdown
    locations = get_locations()
  

    # Generate other plots and components only once
    script_bokeh_plott, div_bokeh_plott = create_bokeh_plott()  # Annual Rainfall Plot remains static

    plotly_heatmap = create_plotly_heatmap()  # Heatmap remains static
    # Default year to show (e.g., 2010)
    # Load the dataset and get the unique years from the data
    unique_years = sorted(dataset['Year'].unique().astype(str))  # Get unique years and sort them
    
    # Default year to show (e.g., 2010)
    year = '2010'
    
    if request.method == 'POST':
        year = request.form.get('year')  # Get the year selected by the user
    
    # Call the plot creation function with the selected year
    script, div = create_rainfall_plot(year)
    # Call the plot creation function
    plot_html = create_correlation_plot()
     # Create the graph
    fig = create_weather_graph(dataset)
    
    fig2 = create_extreme_events_graph(dataset)
    fig3 = create_correlation_heatmap(dataset)

    # Convert the figure to HTML
    graph_html = pio.to_html(fig, full_html=False)
   
     # Create the extreme events graph
    graph_html2 = pio.to_html(fig2, full_html=False)
 # Convert the Plotly figure to HTML to embed in the template
    graph_html3 = pio.to_html(fig3, full_html=False)

    return render_template(
        'index.html',
        script_bokeh_plott=script_bokeh_plott,
        div_bokeh_plott=div_bokeh_plott,
        plotly_heatmap=plotly_heatmap,
        plotly_scatter=plotly_scatter,  # Only the scatter plot updates dynamically
        selected_location=selected_location,
        locations=locations,
        script=script, div=div, 
        selected_year=year, 
        years=unique_years,
        plot_html=plot_html,
        graph_html=graph_html,
        graph_html2=graph_html2,
        graph_html3=graph_html3
    )

@app.route('/update_plot')
def update_plot():
    location = request.args.get('location')
    if not location:
        return jsonify({"error": "No location provided"}), 400

    # Create updated scatter plot for the selected location
    updated_scatter_html = create_plotly_scatter(location)

    # Send back the updated scatter plot HTML
    return jsonify({"scatter_plot_html": updated_scatter_html})



if __name__ == '__main__':
    app.run(debug=True)
