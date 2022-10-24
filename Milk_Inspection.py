import pandas as pd
import numpy as np
import seaborn as sns

import streamlit as st

st.set_page_config(
    page_title="Milk Inspection",
    page_icon="👋",
)

st.write("# Milk Inspection!")

st.sidebar.success("Select a Survey Above.")

st.markdown(
    """
                        Ever had the problem where you just didnt know if your milk was good enough to drink? 
                        Well we got the solution for you, this survey will help you determine if your milk is still drinkable. 
                        The survey will take about 3-5min to fill out and your result will be shown. 
                        The result can contain High - Medium - Low Grades. 
                        When your milk is low grade it means you shouldnt drink it anymore and throw it away, When your milk is medium it
                        will still be drinkable but not as good as high grade milk. 
                        this test is available on all devices
                      
                      
                        The survey is trained with the best trained Artificial intelligence. 
                        We collected a great deal of samples to give to the AI to learn in what grade the milk belongs. 
                        With these samples it is possible to predict the grade of your milk. 
                        You will fill in the survey and the AI will use your answere to predict the quality of milk.
                        You can fill out the survey as many times as you want.
                      
                      
                        Disclaimer: 
                        The survey will not be saved and your data will not be saved to make further progress to our AI. 
                        The AI has a accuracy rate of 95%
                      
    
"""
)  

from PIL import Image
image = Image.open('Melk2.jpg')

st.image(image)