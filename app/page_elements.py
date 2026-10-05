import joblib
import pandas as pd
import streamlit as st


def page_layouts(css_inject, body_sidebar, width_custom, body_title, text_alignment_custom, spec_columns):
    """Generates a page layout with custom CSS, sidebar, title, and specified columns.
    
    Args:
        css_inject (str): Custom CSS to be injected into the HTML.
        body_sidebar (str): Content to be displayed in the sidebar.
        width_custom (int): Width of the sidebar.
        body_title (str): Title to be displayed on the page.
        text_alignment_custom (str): Alignment of the title text (e.g., 'left', 'center', 'right').
        spec_columns (int): Number of columns to create in the layout.
    
    Returns:
        tuple: A tuple containing the following elements:
            - html_custom: The injected HTML with custom CSS.
            - sidebar: The rendered sidebar with the provided content.
            - title: The rendered title of the page.
            - widget_columns: A list of column objects created based on the specified number of columns.
    """
    html_custom = st.html(css_inject)
    sidebar = st.sidebar.markdown(body_sidebar, width=width_custom)
    title = st.title(body_title, text_alignment=text_alignment_custom)
    widget_columns = st.columns(spec_columns)
    return html_custom, sidebar, title, widget_columns

def number_input_widget(description, min_value, max_value, key, step=1, format_custom=None):
    """Generates a number input widget for user interaction.
    
    Args:
        description (str): A brief description of the input field.
        min_value (int or float): The minimum value that can be input.
        max_value (int or float): The maximum value that can be input.
        key (str): A unique key for the widget.
        step (int or float, optional): The increment step for the input value. Defaults to 1.
        format_custom (str, optional): A format string for displaying the value. Defaults to None.
    
    Returns:
        float or int: The value entered by the user.
    """
    return st.number_input(description, min_value, max_value,
                           step=step,
                           format=format_custom,
                           key=key)

def slider_widget(description, min_value, max_value, key, step=1, format_custom=None, integer=False):
    """Creates a slider widget for user input.
    
    Args:
        description (str): A brief description of the slider's purpose.
        min_value (int or float): The minimum value the slider can take.
        max_value (int or float): The maximum value the slider can take.
        key (str): A unique key for the widget, used for state management.
        step (int or float, optional): The increment step for the slider. Defaults to 1.
        format_custom (str, optional): A format string for displaying the slider value. Defaults to None.
        integer (bool, optional): If True, the slider will only allow integer values. Defaults to False.
    
    Returns:
        float or int: The value selected by the user on the slider.
    """
    value_midpoint = (min_value + max_value) / 2
    if integer:
        return st.slider(description, int(min_value), int(max_value),
                         value=int(value_midpoint),
                         step=step,
                         key=key)
    else:
        return st.slider(description, min_value, max_value,
                         value=value_midpoint,
                         step=step,
                         format=format_custom,
                         key=key)

def widget_session_state(column, value):
    """Generates an HTML display based on the given column and value.
    
    Args:
        column (str): The name of the column.
        value (float): The value from user input.
    
    Returns:
        None: This function does not return a value; it updates the session state with the provided column and value.
    """
    if column not in st.session_state:
        st.session_state[column] = value

def widget_column_container(columns, width_custom, model_path, index_first):
    """Generates an HTML container displaying estimated fuel economy.
    
    Args:
        columns (list): A list of columns to be displayed in the container.
        width_custom (str): A custom width for the container.
        model_path (str): The file path to the model used for predictions.
        index_first (int): The index of the first column to be processed.
    
    Returns:
        str: An HTML string containing the estimated fuel economy formatted for display.
    """
    user_input = {column: st.session_state[column] for column in columns}
    input_df = pd.DataFrame([user_input])
    key_container = "container"
    with st.container(border=True, key=key_container), st.spinner(width=width_custom):
        model = joblib.load(model_path)
        prediction = model.predict(input_df)[index_first]
        prediction_display = f"""
        <p style='margin: 0; text-align: center;'>
            Estimated fuel economy: <font color='red'>{prediction:.2f} MPG</font>
        </p>
        """
        return st.html(prediction_display)