import streamlit as st

st.markdown("<style> div { text-align: center } </style>", unsafe_allow_html=True)
st.subheader('EXTRACT DATA')


with st.form(key='form1'):
    uploaded_file = st.file_uploader("Choose a csv file", type="csv")
    submit_button2 = st.form_submit_button(label='Aceptar')

st.divider()
