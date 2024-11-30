
import streamlit as st

# Set the title of the app
st.title("Progress Bar Example")

# Define the values
label = "Aspect"

# Create the progress bar
# Display the label and value on the same line
col1, col2 = st.columns([3, 1])  # Create two columns
with col1:
    st.write(label)
with col2:
    st.write('Cleanliness')
progress = st.progress(1 / 1)

# Optionally, you can add some other content or functionality below
st.write("Your content goes here.")
