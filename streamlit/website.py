import streamlit as st
import pandas as pd
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import os
import numpy as np

# local imports
os.chdir("../src")
from streamlit_mlp_api import streamlit_mlp_static_api, streamlit_mlp_inference_api
from transform_canvas_image import transform_canvas_image
os.chdir("../streamlit")


# Initialize MLP Models for later use
mlp_folder_path = "../models/MLP"
folders = [f for f in os.listdir(mlp_folder_path) if os.path.isdir(os.path.join(mlp_folder_path, f))]
folders = folders[1:] # Remove "DEMO"
mlp_model_path = folders.copy()
mlp_model_names = [f"HiddenLayer_{f.split('_')[1]}_HiddenSize_{f.split('_')[2]}_TotalParams_{f.split('_')[3]}_Result_{f.split('_')[4]}" for f in folders]

# Your Streamlit code goes here
st.set_page_config(layout="wide")

# Title
st.markdown("<h1 style='text-align: center;'>Getting to know Kolmogorov-Arnold Networks</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Select a page", ["Home", "Model Visualization", "Inference Test Page"], index=0)

# Initialize session state
if "predict_clicked" not in st.session_state:
    st.session_state.predict_clicked = False
if "reset_clicked" not in st.session_state:
    st.session_state.reset_clicked = False
if "updating" not in st.session_state:
    st.session_state.updating = False
if "mlp_chart_data" not in st.session_state:
    st.session_state.mlp_chart_data = pd.DataFrame()
if "mlp_image" not in st.session_state:
    st.session_state.mlp_image = None
if "kan_chart_data" not in st.session_state:
    st.session_state.kan_chart_data = pd.DataFrame()
if "kan_image" not in st.session_state:
    st.session_state.kan_image = None
if "image_to_predict" not in st.session_state:
    st.session_state.image_to_predict = None
if "mlp_standard_vis" not in st.session_state:
    st.session_state.mlp_standard_vis = None
if "kan_standard_vis" not in st.session_state:
    st.session_state.kan_standard_vis = None
if "mlp_path" not in st.session_state:
    st.session_state.mlp_path = None
if "kan_path" not in st.session_state:
    st.session_state.kan_path = None

# Homepage
if selection == "Home":
    st.write("<div style='text-align: center; margin-top: 50px;'>Welcome to the Kolmogorov-Arnold Networks app! This app allows you to explore and visualize the concepts of Kolmogorov-Arnold Networks. To get started, please choose a page from the sidebar to navigate through the app.</div>", unsafe_allow_html=True)

# Model Visualization
# This page is for visualizing the models. The user can select a model from the dropdown menu and see the visualization of the model architecture.
elif selection == "Model Visualization":

    col1, col2, col3 = st.columns([1, 0.1, 1])

    # Left Column
    with col1:
        
        # Title
        st.markdown("<h2 style='text-align: center; margin-top: 50px;'>MLP</h2>", unsafe_allow_html=True)

        # Dropdown menu for selecting the MLP model
        mlp_model = st.selectbox("Select a model", mlp_model_names, key="mlp_model", help="Select a model", index=0)

        # Get the path of the selected model
        mlp_index = mlp_model_names.index(mlp_model)
        mlp_path = "../models/MLP/" + mlp_model_path[mlp_index] + "/model.pth"

        # If the selected model is different from the previous model, update the session state
        if mlp_path != st.session_state.get("mlp_path", None):
            st.session_state.mlp_path = mlp_path
        st.session_state.mlp_standard_vis = streamlit_mlp_static_api(st.session_state.mlp_path)

        # Display the visualization of the model
        st.image(st.session_state.mlp_standard_vis)

    # Middle Column
    with col2:

        # Just a Divider
        st.markdown(
            '''
                <div class="divider-container">
                    <div class="divider-vertical-line"></div>
                </div>
                <style>
                    .divider-container {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100%;
                    }
                    .divider-vertical-line {
                        border-left: 5px solid rgba(49, 51, 63, 0.6);
                        height: 45em; 
                    }
                </style>
            ''', 
            unsafe_allow_html=True
        )

    # Right Column
    with col3:

        # Title
        st.markdown("<h2 style='text-align: center; margin-top: 50px;'>KAN</h2>", unsafe_allow_html=True)

        # Dropdown menu for selecting the KAN model
        kan_model = st.selectbox("Select a model", ["Placeholder1", "Placeholder2", "Placeholder3"], key="kan_model", help="Select a model")

        match kan_model:
            case "Placeholder1":
                st.image("https://images.prismic.io/encord/11b9026c-edc4-4d23-b6f3-09bd0ede3e28_image+%2835%29+2.jpg?auto=compress%2Cformat&fit=max&w=906&h=638", caption="Placeholder for the model visualization", use_column_width=True)
            case "Placeholder2":
                st.write("Placeholder for model 2")
            case "Placeholder3":
                st.write("Placeholder for model 3")

# Inference Test Page
elif selection == "Inference Test Page":

    # 3 Columns for visualization of the inference results and the drawing field 
    col1, col2, col3 = st.columns(3)

    # Middle Column
    with col2:

        # Title
        st.markdown("<h2 style='text-align: center; margin-top: 50px;'>Draw</h2>", unsafe_allow_html=True)

        # Creating the drawing canvas
        canvas_width = canvas_height = 28 * 10
        with st.container():
            tmp1, tmp2, tmp3 = st.columns([1, 4, 1])
            with tmp2:
                drawing = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=40,
                    stroke_color="#000000",
                    background_color="#eee",
                    background_image=None,
                    update_streamlit=True,
                    height=canvas_height,
                    width=canvas_width,
                    drawing_mode="freedraw",
                    point_display_radius=0,
                    display_toolbar=True,
                    key="full_app",
                )
            
            # Buttons for prediction and reset
            tmp1, tmp2, tmp3, tmp4, tmp5 = st.columns([0.5, 1, 1, 1, 0.5])
            with tmp2:
                predict_button = st.button("Predict")
                if predict_button:
                    # Predict_clicked changes the displayed content permanently | "updating" is used to update the displayed content
                    st.session_state.predict_clicked = True
                    st.session_state.updating = True
            with tmp4:
                reset_button = st.button("Reset")
                if reset_button:
                    # Variable to reset the displayed content to the initial state
                    st.session_state.reset_clicked = True

    # Left Column
    with col1:

        # Title
        st.markdown("<h2 style='text-align: center; margin-top: 50px;'>MLP</h2>", unsafe_allow_html=True)

        # Dropdown menu for selecting the MLP model
        mlp_model = st.selectbox("Select a model", mlp_model_names, key="mlp_model", help="Select a model", index=0)
        mlp_index = mlp_model_names.index(mlp_model)
        mlp_path = "../models/MLP/" + mlp_model_path[mlp_index] + "/model.pth"

    # Right Column
    with col3:

        # Title
        st.markdown("<h2 style='text-align: center; margin-top: 50px;'>KAN</h2>", unsafe_allow_html=True)

        # Dropdown menu for selecting the KAN model
        kan_model = st.selectbox("Select a model", ["Test"], key="kan_model", help="Select a model", index=0)
    
    # Displaying the architecture visualization of the models for the left and right column
    with col1:
        mlp_diagram = st.empty()
        mlp_standard_vis = st.empty()

        # Updating the visualization of the MLP model
        if not st.session_state.get("predict_clicked", False) or st.session_state.get("reset_clicked", False):
            #st.write(mlp_path, st.session_state.get("mlp_path", None))
            if mlp_path != st.session_state.get("mlp_path", None):
                st.session_state.mlp_path = mlp_path
            st.session_state.mlp_standard_vis = streamlit_mlp_static_api(st.session_state.mlp_path)
            st.image(st.session_state.mlp_standard_vis)

    with col3:
        kan_diagram = st.empty()
        kan_standard_vis = st.empty()
        
        # Updating the visualization of the KAN model
        if not st.session_state.get("predict_clicked", False) or st.session_state.get("reset_clicked", False):
            if kan_model != st.session_state.get("kan_model", None):
                st.session_state.kan_model = kan_model
                #st.session_state.kan_standard_vis = streamlit_kan_static_api(kan_model)
            #st.image(st.session_state.kan_standard_vis)
            st.image("https://media.tenor.com/patKcXgljGYAAAAe/cursed-cat.png")


    # Displayed content changes permanently when the predict button is clicked
    if st.session_state.get("predict_clicked", False):
        # Checking if the reset button is clicked
        if not st.session_state.get("reset_clicked", False):
            
            # Checking if the displayed content should be updated
            if st.session_state.get("updating", False):
                
                # Transforming the drawing to an image that fits for the models
                st.session_state.image_to_predict = transform_canvas_image(drawing)

                # MLP Inference
                # Getting the prediction results and the visualization of the inference
                pred_df, img = streamlit_mlp_inference_api(mlp_path,  st.session_state.image_to_predict)

                # Updating the session states for the chart data and the image
                st.session_state.mlp_chart_data = pred_df
                st.session_state.mlp_image = img
                
                # KAN Inference
                # ----------------------------------------------------------------------------------
                match kan_model:
                    case "Test":
                        chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
                        img = Image.open("pics/test_model.png")
                # ----------------------------------------------------------------------------------
                st.session_state.kan_chart_data = chart_data
                st.session_state.kan_image = img
                
                
            with tmp3:
                # Displaying the converted image for the user
                st.image(st.session_state.image_to_predict, caption="Converted image")

            # Displaying the bar charts (probability distribution) for the MLP and KAN models
            mlp_diagram.bar_chart(st.session_state.mlp_chart_data, y_label="Probability", x_label="Digit")
            kan_diagram.bar_chart(st.session_state.kan_chart_data, y_label="Probability", x_label="Digit")

            # Displaying the architecture for the MLP and KAN models
            with col1:
                st.image(st.session_state.mlp_image, caption="Inference Visualization", use_column_width=True)
            
            with col3:
                st.image(st.session_state.kan_image, caption="Inference Visualization", use_column_width=True)

            # Resetting the updating variable -> With the next click on the predict button, the displayed content will be updated
            st.session_state.updating = False

        # If the reset button is clicked, the displayed content will be reset to the initial state
        else:
            # Resetting the session states, so no chart data and images are displayed, just the initial state as in the visualization page
            st.session_state.reset_clicked = False
            st.session_state.predict_clicked = False