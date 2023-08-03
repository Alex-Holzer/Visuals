import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_bar_chart(dataframe: pd.DataFrame, x_column: str, y_column: str, colormap: str = 'viridis',
                   title: str = None, horizontal: bool = False, show_mean_median: bool = False,
                   show_number: bool = False) -> None:
    """
    Generate a bar chart using the provided DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data for the bar chart.
        x_column (str): The name of the column to be used for the x-axis of the chart.
        y_column (str): The name of the column to be used for the y-axis of the chart.
        colormap (str, optional): The name of the colormap to use for coloring the bars.
                                  Default is 'viridis'.
        title (str, optional): The title for the chart. If None, the chart will have no title.
        horizontal (bool, optional): If True, the chart will be displayed as a horizontal bar chart.
                                     Default is False (vertical bar chart).
        show_mean_median (bool, optional): If True, the mean and median values will be displayed in the top-right
                                           corner of the chart with a frame. Default is False.
        show_number (bool, optional): If True, the values of each bar will be displayed at the top of the bar.
                                      Default is False.

    Returns:
        None: The function displays the bar chart directly without returning anything.
    """

    # Step 1: Sort the DataFrame by the y_column in descending order
    df = dataframe.sort_values(by=y_column, ascending=False)

    # Create the bar plot using plotly
    fig = go.Figure()

    if horizontal:
        fig.add_trace(go.Bar(y=df[x_column], x=df[y_column], orientation='h', marker=dict(
            color=df[y_column], colorscale=colormap)))
        fig.update_xaxes(title_text=y_column)
        fig.update_yaxes(title_text=x_column)
        # Invert y-axis for horizontal bar chart
        fig.update_layout(yaxis=dict(autorange="reversed"))
        # Display values at the top of each bar if show_number is True
        if show_number:
            for index, value in enumerate(df[y_column]):
                fig.add_annotation(
                    x=value,
                    y=df[x_column].iloc[index],
                    text=f'{value:.2f}',
                    showarrow=False,
                    xshift=20,
                    font=dict(size=12),
                )

    else:
        fig.add_trace(go.Bar(x=df[x_column], y=df[y_column], marker=dict(
            color=df[y_column], colorscale=colormap)))
        fig.update_xaxes(title_text=x_column)
        fig.update_yaxes(title_text=y_column)
        # Display values at the top of each bar if show_number is True
        if show_number:
            for index, value in enumerate(df[y_column]):
                fig.add_annotation(
                    x=df[x_column].iloc[index],
                    y=value,
                    text=f'{value:.2f}',
                    showarrow=False,
                    yshift=5,
                    font=dict(size=12),
                )

    # Display mean and median values if show_mean_median is True
    if show_mean_median:
        # Calculating mean and median
        mean_value = df[y_column].mean()
        median_value = df[y_column].median()

        # Adding mean and median values in the top-right corner of the chart with a frame
        text_to_display = f'Mean: {mean_value:.2f}<br>Median: {median_value:.2f}'
        fig.add_annotation(
            x=0.99,
            y=0.99,
            xref='paper',
            yref='paper',
            text=text_to_display,
            showarrow=False,
            font=dict(size=12),
        )

    if title:
        fig.update_layout(title=dict(text=title, x=0.5,
                          font=dict(size=18, color='#333333')))

    # Display the plot
    fig.update_layout(margin=dict(t=60, b=60, l=60, r=60))
    fig.show()


def plot_group_bar_chart(dataframe: pd.DataFrame, x_column: str, values_column: str, group_column: str,
                         title: str = None, colormap: str = 'viridis', horizontal: bool = False) -> None:
    """
    Generate a grouped bar chart using the provided DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data for the grouped bar chart.
        x_column (str): The name of the column to be used for the x-axis of the chart.
        values_column (str): The name of the column to be used for the values of the grouped bars.
        group_column (str): The name of the column to be used for grouping the bars.
        title (str, optional): The title for the chart. If None, the chart will have no title.
        colormap (str, optional): The name of the colormap to use for coloring the bars.
                                  Default is 'viridis'.
        horizontal (bool, optional): If True, the chart will be displayed as a horizontal grouped bar chart.
                                     If False, the chart will be displayed as a vertical grouped bar chart (default).

    Returns:
        None: The function displays the grouped bar chart directly without returning anything.
    """

    data_to_plot = dataframe.pivot(
        index=group_column, columns=x_column, values=values_column)

    # Create the grouped bar chart using Plotly
    fig = go.Figure()

    for col in data_to_plot.columns:
        if horizontal:
            fig.add_trace(go.Bar(y=data_to_plot.index, x=data_to_plot[col], name=col, orientation='h',
                                 marker=dict(colorscale=colormap)))
        else:
            fig.add_trace(go.Bar(x=data_to_plot.index, y=data_to_plot[col], name=col,
                                 marker=dict(colorscale=colormap)))

    # Set axis labels and title
    fig.update_layout(title_text=title, xaxis_title=group_column,
                      yaxis_title=values_column)

    # Show legend
    fig.update_layout(showlegend=True)

    # Display the plot
    fig.show()


def plot_stacked_bar_chart(dataframe: pd.DataFrame, x_column: str, values_column: str, group_column: str,
                           title: str = None, colormap: str = 'viridis', horizontal: bool = False) -> None:
    """
    Generate a stacked bar chart using the provided DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data for the stacked bar chart.
        x_column (str): The name of the column to be used for the x-axis of the chart.
        values_column (str): The name of the column to be used for the values of the stacked bars.
        group_column (str): The name of the column to be used for grouping the stacked bars.
        title (str, optional): The title for the chart. If None, the chart will have no title.
        colormap (str, optional): The name of the colormap to use for coloring the bars.
                                  Default is 'viridis'.
        horizontal (bool, optional): If True, the chart will be displayed as a horizontal stacked bar chart.
                                     If False, the chart will be displayed as a vertical stacked bar chart (default).

    Returns:
        None: The function displays the stacked bar chart directly without returning anything.
    """

    data_to_plot = dataframe.pivot(
        index=x_column, columns=group_column, values=values_column)

    # Create the stacked bar chart using Plotly
    fig = go.Figure()

    for col in data_to_plot.columns:
        if horizontal:
            fig.add_trace(go.Bar(y=data_to_plot.index, x=data_to_plot[col], name=col, orientation='h',
                                 marker=dict(colorscale=colormap)))
        else:
            fig.add_trace(go.Bar(x=data_to_plot.index, y=data_to_plot[col], name=col,
                                 marker=dict(colorscale=colormap)))

    # Set axis labels and title
    fig.update_layout(title_text=title, xaxis_title=x_column,
                      yaxis_title=values_column)

    # Set barmode to 'stack' to stack the bars vertically
    fig.update_layout(barmode='stack')

    # Show legend
    fig.update_layout(showlegend=True)

    # Display the plot
    fig.show()


def plot_line_chart(dataframe: pd.DataFrame, x_column: str, y_columns: list, colormap: str = 'tab20',
                    title: str = None, xlabel: str = None, ylabel: str = None) -> None:
    """
    Generate a line chart with multiple lines using the provided DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data for the line chart.
        x_column (str): The name of the column to be used for the x-axis of the chart.
        y_columns (list): The names of the columns to be used for the y-axis of the chart (multiple lines).
        colormap (str, optional): The name of the colormap to use for coloring the lines.
                                  Default is 'tab20'.
        title (str, optional): The title for the chart. If None, the chart will have no title.
        xlabel (str, optional): The label for the x-axis. If None, the x-axis will have no label.
        ylabel (str, optional): The label for the y-axis. If None, the y-axis will have no label.

    Returns:
        None: The function displays the line chart directly without returning anything.
    """

    # Create the line chart using Plotly
    fig = go.Figure()

    # Add each line to the figure
    for column in y_columns:
        fig.add_trace(go.Scatter(
            x=dataframe[x_column], y=dataframe[column], mode='lines+markers', name=column))

    # Update layout for axis labels and title
    if xlabel:
        fig.update_xaxes(title_text=xlabel)
    if ylabel:
        fig.update_yaxes(title_text=ylabel)
    if title:
        fig.update_layout(title_text=title)

      # Set barmode to 'stack' to stack the bars vertically
   # fig.update_layout(barmode='stack')

    # Display the plot
    fig.show()


def plot_combined_line_and_bar_chart(dataframe: pd.DataFrame, x_column: str, line_columns: list,
                                     bar_column: str, title: str = None, xlabel: str = None,
                                     ylabel_line: str = None, ylabel_bar: str = None):
    """
    Generate a chart that combines a line chart and a bar chart using the provided DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data for the charts.
        x_column (str): The name of the column to be used for the x-axis of the chart.
        line_columns (list): The names of the columns to be used for the y-axis of the line chart (multiple lines).
        bar_column (str): The name of the column to be used for the bar chart.
        title (str, optional): The title for the chart. If None, the chart will have no title.
        xlabel (str, optional): The label for the x-axis. If None, the x-axis will have no label.
        ylabel_line (str, optional): The label for the y-axis of the line chart. If None, the y-axis will have no label.
        ylabel_bar (str, optional): The label for the y-axis of the bar chart. If None, the y-axis will have no label.

    Returns:
        None: The function displays the combined chart directly without returning anything.
    """

    # Create a subplot with two rows and one column
    fig = make_subplots(rows=2, cols=1, subplot_titles=(
        "Line Chart", "Bar Chart"))

    # Add each line to the line chart subplot
    for i, column in enumerate(line_columns):
        fig.add_trace(go.Scatter(
            x=dataframe[x_column], y=dataframe[column], mode='lines+markers', name=column), row=1, col=1)

    # Add the bar chart to the bar chart subplot
    fig.add_trace(go.Bar(
        x=dataframe[x_column], y=dataframe[bar_column], name=bar_column), row=2, col=1)

    # Update layout for axis labels and title
    if xlabel:
        fig.update_xaxes(title_text=xlabel, row=1, col=1)
    if ylabel_line:
        fig.update_yaxes(title_text=ylabel_line, row=1, col=1)
    if ylabel_bar:
        fig.update_yaxes(title_text=ylabel_bar, row=2, col=1)
    if title:
        fig.update_layout(title_text=title)

    # Display the plot
    fig.show()


def plot_scatter_plot(dataframe: pd.DataFrame, x_column: str, y_column: str, colormap: str = 'Viridis',
                      title: str = None, xlabel: str = None, ylabel: str = None, color_column: str = None) -> None:
    """
    Generate a scatter plot using the provided DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data for the scatter plot.
        x_column (str): The name of the column to be used for the x-axis of the chart.
        y_column (str): The name of the column to be used for the y-axis of the chart.
        colormap (str, optional): The name of the colormap to use for coloring the data points.
                                  Default is 'Viridis'.
        title (str, optional): The title for the chart. If None, the chart will have no title.
        xlabel (str, optional): The label for the x-axis. If None, the x-axis will have no label.
        ylabel (str, optional): The label for the y-axis. If None, the y-axis will have no label.
        color_column (str, optional): The name of the column to be used for coloring the data points based on
                                      distinct values in this column.

    Returns:
        None: The function displays the scatter plot directly without returning anything.
    """

    # Create the scatter plot using Plotly Express
    fig = px.scatter(dataframe, x=x_column, y=y_column, color=color_column, color_continuous_scale=colormap,
                     labels={x_column: xlabel, y_column: ylabel}, title=title)

    # Customize legend title
    if color_column:
        fig.update_layout(legend_title_text='Categories')

    # Display the plot
    fig.show()


def plot_bubble_chart(dataframe: pd.DataFrame, x_column: str, y_column: str, size_column: str, colormap: str = 'Viridis',
                      title: str = None, xlabel: str = None, ylabel: str = None, color_column: str = None) -> None:
    """
    Generate a bubble chart using the provided DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data for the bubble chart.
        x_column (str): The name of the column to be used for the x-axis of the chart.
        y_column (str): The name of the column to be used for the y-axis of the chart.
        size_column (str): The name of the column to be used for the size of the bubbles.
        colormap (str, optional): The name of the color scale to use for coloring the data points.
                                  Default is 'Viridis'.
        title (str, optional): The title for the chart. If None, the chart will have no title.
        xlabel (str, optional): The label for the x-axis. If None, the x-axis will have no label.
        ylabel (str, optional): The label for the y-axis. If None, the y-axis will have no label.
        color_column (str, optional): The name of the column to be used for coloring the data points based on
                                      distinct values in this column.

    Returns:
        None: The function displays the bubble chart directly without returning anything.
    """

    # Create the bubble chart using Plotly Express
    fig = px.scatter(dataframe, x=x_column, y=y_column, size=size_column, color=color_column,
                     color_continuous_scale=colormap, hover_name=color_column)

    # Adding labels and title
    if xlabel:
        fig.update_xaxes(title_text=xlabel)
    if ylabel:
        fig.update_yaxes(title_text=ylabel)
    if title:
        fig.update_layout(title_text=title)

    # Display the plot
    fig.show()


def plot_heatmap(dataframe: pd.DataFrame, x_column: str, y_column: str, values_column: str, colormap: str = 'Viridis',
                 title: str = None, xlabel: str = None, ylabel: str = None):
    """
    Generate a heatmap using the provided DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data for the heatmap.
        x_column (str): The name of the column to be used for the x-axis of the heatmap.
        y_column (str): The name of the column to be used for the y-axis of the heatmap.
        values_column (str): The name of the column to be used for the values of the heatmap cells.
        colormap (str, optional): The name of the colormap to use for coloring the heatmap cells.
                                  Default is 'Viridis'.
        title (str, optional): The title for the heatmap. If None, the heatmap will have no title.
        xlabel (str, optional): The label for the x-axis. If None, the x-axis will have no label.
        ylabel (str, optional): The label for the y-axis. If None, the y-axis will have no label.

    Returns:
        None: The function displays the heatmap directly without returning anything.
    """

    # Pivot the DataFrame to get the data in a suitable format for the heatmap
    pivot_df = dataframe.pivot(
        index=y_column, columns=x_column, values=values_column)

    # Create the heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(z=pivot_df.values, x=pivot_df.columns, y=pivot_df.index,
                                    colorscale=colormap, zmin=pivot_df.min().min(), zmax=pivot_df.max().max()))

    # Adding labels and title
    if xlabel:
        fig.update_xaxes(title_text=xlabel)
    if ylabel:
        fig.update_yaxes(title_text=ylabel)
    if title:
        fig.update_layout(title_text=title)

    # Display the plot
    fig.show()


def plot_correlation_heatmap(dataframe: pd.DataFrame, colormap: str = 'RdBu', title: str = None):
    """
    Generate a correlation heatmap using the provided DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data for the correlation heatmap.
        colormap (str, optional): The name of the colorscale to use for coloring the heatmap cells.
                                  Default is 'RdBu'.
        title (str, optional): The title for the heatmap. If None, the heatmap will have no title.

    Returns:
        None: The function displays the correlation heatmap directly without returning anything.
    """

    # Calculate the correlation matrix
    corr_matrix = dataframe.corr()

    # Create the heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
                                    colorscale=colormap, zmin=-1, zmax=1, reversescale=True))

    # Adding labels and title
    fig.update_layout(
        title=title,
        xaxis=dict(tickfont=dict(size=12), tickangle=-45),
        yaxis=dict(tickfont=dict(size=12)),
        coloraxis_colorbar=dict(title='Correlation',
                                titlefont=dict(size=14), ticks='inside')
    )

    # Display the plot
    fig.show()


def plot_histogram(dataframe: pd.DataFrame, column: str, bins: int = 10,
                   colormap: str = 'Viridis', title: str = None, xlabel: str = None, ylabel: str = None):
    """
    Generate a histogram using the provided DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data for the histogram.
        column (str): The name of the column to be used for the histogram.
        bins (int, optional): The number of bins to use for the histogram. Default is 10.
        colormap (str, optional): The name of the colorscale to use for coloring the bars.
                                  Default is 'Viridis'.
        title (str, optional): The title for the histogram. If None, the histogram will have no title.
        xlabel (str, optional): The label for the x-axis. If None, the x-axis will have no label.
        ylabel (str, optional): The label for the y-axis. If None, the y-axis will have no label.

    Returns:
        None: The function displays the histogram directly without returning anything.
    """

    # Create the histogram using Plotly
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=dataframe[column], nbinsx=bins, marker_colorscale=colormap))

    # Adding labels and title
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis=dict(title_font=dict(size=14)),
        yaxis=dict(title_font=dict(size=14))
    )

    # Display the plot
    fig.show()


def plot_stacked_histogram(dataframe: pd.DataFrame, x_column: str, color_column: str,
                           bins: int = 10, title: str = None, xlabel: str = None, ylabel: str = None):
    """
    Generate a stacked histogram using the provided DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data for the histogram.
        x_column (str): The name of the column to be used for the x-axis of the histogram.
        color_column (str): The name of the column to be used for coloring the bars and creating the stacks.
        bins (int, optional): The number of bins to use for the histogram. Default is 10.
        title (str, optional): The title for the histogram. If None, the histogram will have no title.
        xlabel (str, optional): The label for the x-axis. If None, the x-axis will have no label.
        ylabel (str, optional): The label for the y-axis. If None, the y-axis will have no label.

    Returns:
        None: The function displays the stacked histogram directly without returning anything.
    """

    # Create the stacked histogram using Plotly Express
    fig = px.histogram(dataframe, x=x_column,
                       color=color_column, nbins=bins, text_auto=True)

    # Customize the layout
    if title:
        fig.update_layout(title_text=title, title_font=dict(
            size=16, color='#333333'))

    if xlabel:
        fig.update_xaxes(title_text=xlabel, title_font=dict(
            size=14, color='#333333'))

    if ylabel:
        fig.update_yaxes(title_text=ylabel, title_font=dict(
            size=14, color='#333333'))

    # Show the plot
    fig.show()


def plot_box_plot(dataframe: pd.DataFrame, x_column: str, y_column: str,
                  title: str = None, xlabel: str = None, ylabel: str = None,
                  point: str = None):
    """
    Generate a box plot using the provided DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data for the box plot.
        x_column (str): The name of the column to be used for the x-axis of the box plot.
        y_column (str): The name of the column to be used for the y-axis of the box plot.
        title (str, optional): The title for the box plot. If None, the box plot will have no title.
        xlabel (str, optional): The label for the x-axis. If None, the x-axis will have no label.
        ylabel (str, optional): The label for the y-axis. If None, the y-axis will have no label.
        point (str, optional): The points of the DataFrame. If all, then points are visible.  
    Returns:
        None: The function displays the box plot directly without returning anything.
    """

    # Create the box plot using Plotly Express
    fig = px.box(dataframe, x=x_column, y=y_column, points=point)

    # Customize the layout
    if title:
        fig.update_layout(title_text=title, title_font=dict(
            size=16, color='#333333'))

    if xlabel:
        fig.update_xaxes(title_text=xlabel, title_font=dict(
            size=14, color='#333333'))

    if ylabel:
        fig.update_yaxes(title_text=ylabel, title_font=dict(
            size=14, color='#333333'))

    # Show the plot
    fig.show()


def plot_treemap(dataframe: pd.DataFrame, path_column: str, values_column: str,
                 title: str = None, color_column: str = None):
    """
    Generate a treemap using the provided DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data for the treemap.
        path_column (str): The name of the column to be used for the hierarchy in the treemap.
        values_column (str): The name of the column to be used for the values in the treemap.
        title (str, optional): The title for the treemap. If None, the treemap will have no title.
        color_column (str, optional): The name of the column to be used for coloring the treemap cells.
                                      If None, the treemap cells will have the same color.

    Returns:
        None: The function displays the treemap directly without returning anything.
    """

    # Create the treemap using Plotly Express
    fig = px.treemap(dataframe, path=[path_column], values=values_column, color=color_column,
                     color_continuous_scale='Blues', hover_data=[values_column])

    # Customize the layout
    if title:
        fig.update_layout(title_text=title, title_font=dict(
            size=16, color='#333333'))

    # Show the plot
    fig.show()
