import streamlit as st
import requests
import time

# Title of the app
st.title("Aspect-Based Sentiment Analysis")

# Initialize session state for dataset selection
if 'dataset_selected' not in st.session_state:
    st.session_state['dataset_selected'] = False
    st.session_state['dataset'] = None

# Sidebar for dataset selection
if not st.session_state['dataset_selected']:
    dataset_options = ["Hotel", "Restaurant"]
    dataset = st.sidebar.selectbox("Select Dataset", dataset_options)
    if st.sidebar.button("Confirm Selection"):
        st.session_state['dataset_selected'] = True
        st.session_state['dataset'] = dataset
else:
    dataset = st.session_state['dataset']
    st.sidebar.write(f"Selected Dataset: {dataset}")
    st.sidebar.write("To change the dataset, please restart the app.")

# Navigation options
if st.session_state['dataset_selected']:
    # Create two columns for horizontal layout
    col1, col2 = st.columns(2)

    with col1:
        option = st.radio("Choose an option:", ["Real Performance", "Sentiment in Test Set"])

    with col2:
        id_option = st.radio("Choose an option:", ["PhoBertLarge"])

    if option == "Sentiment in Test Set":
        # Display the best result in the test set based on the selected dataset
        st.write(f"### Best Result in {dataset} Test Set")

        # Load result text file directly
        file_path = f"predictions/{dataset}-{id_option}.txt"
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                file_contents = file.read()
                st.write("### Result Text File:")
                st.text(file_contents)
        except FileNotFoundError:
            st.error(f"Result file for {dataset} dataset not found.")

    elif option == "Real Performance":
        # Text area for user input
        input_text = st.text_area("Enter text for sentiment analysis:")

        # Button to trigger analysis
        if st.button("Analyze"):
            if input_text:
                with st.spinner("Analyzing..."):
                    # Call your backend API
                    response = requests.post('http://127.0.0.1:8000/analyze', json={"text": input_text, "dataset": dataset, "id_option": id_option})
                    time.sleep(1)  # Simulate a delay for demonstration purposes

                    if response.status_code == 200:
                        results = response.json()
                        st.success("Analysis complete!")
                        st.write("### Sentiment Analysis Results:")
                        for result in results:
                            aspect = result['Aspect']
                            polarity = result['Polarity']
                            if polarity is not None:
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**Aspect:** {aspect}")
                                with col2:
                                    st.write(f"**Sentiment:** {polarity}")
                                if polarity == "positive":
                                    st.progress(1.0)
                                elif polarity == "neutral":
                                    st.progress(0.5)
                                elif polarity == "negative":
                                    st.progress(0.0)
                    else:
                        st.error("Error in API call")
            else:
                st.warning("Please enter text for analysis")

# Footer
st.markdown("---")
st.markdown("Developed by [Your Name](https://your-linkedin-profile)")