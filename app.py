import time
import pandas as pd
import numpy as np
import streamlit as st
from models import Bert, SimpleTransformer, BertClassifier
import re


st.set_page_config(
    page_title="Tunisian Dialect Sentiement Analysis",
    page_icon="🧾️",
    layout="wide"
)

def _max_width_():
    max_width_str = f"max-width: 100%;"
    st.markdown(
        f"""
    <style>
    {{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()


c30, c31 = st.columns([1,5])

with c30:
    st.image("TunBERT.png", width=200)
with c31:
    st.title("""Tunisian Dialect Sentiement Analysis """)
    st.header("")

with st.expander("ℹ️ - About this Application", expanded=False):

    st.write(
        """     
-   The Tunisian Dialect Sentiement Analysis app is an easy-to-use interface !
-   On social media, Arabic speakers tend to express themselves in their own local dialect. To do so, Tunisians use Tunisian Arabizi, where the Latin alphabet is supplemented with numbers. However, annotated datasets for Arabizi are limited.

-   Such solutions could be used by banking, insurance companies, or social media influencers to better understand and interpret a product's audience and their reactions.
	    """
    )

    st.markdown("")

st.markdown("")
st.markdown("### 📖Dataset Used for Training the Models")

df = pd.read_csv("Train.csv")
st.dataframe(df.sample(10), width=10000,height=500)

st.markdown("")
st.markdown("### 🤖 Analyse Comments in Tunisian Dialect ")
with st.form(key="my_form"):


    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    with c1:
        ModelType = st.selectbox(
            "Choose your model",
            ["None", "BERT Multilingual", "BERT Camembert (French)", "Arabic BERT", "Ensemble Three Models"],
            help="You can choose which model you want to use to analyse your comment",
        )

        if ModelType == "BERT Multilingual":
            model_name = "distilbert"
            model = SimpleTransformer('distilbert', 'ppp_gl3/outputs', n_epochs=2,lr=2e-4,seq_len=150,train_batch_size=160)

        elif ModelType == "BERT Camembert (French)":

            model_name = "camembert-base"
            model = Bert("camembert-base", "ppp_gl3/fr_best.pt")
        
        elif ModelType == "Arabic BERT":

            model_name = "moha/mbert_ar_c19"
            model = Bert('moha/mbert_ar_c19' ,"ppp_gl3/ar_best.pt")

        else :

            model_multi = SimpleTransformer('distilbert', 'ppp_gl3/outputs', n_epochs=2,lr=2e-4,seq_len=150,train_batch_size=160)
            model_fr = Bert("camembert-base", "ppp_gl3/fr_best.pt")
            model_ar = Bert('moha/mbert_ar_c19' ,"ppp_gl3/ar_best.pt")
            list_model = [model_multi, model_fr, model_ar]

        with c2:
            doc = st.text_area(
                "Paste your comment below",
                height=250,
            )

            MAX_WORDS = 256
            
        submit_button = st.form_submit_button(label="✨ Analyse!")
        
        if submit_button :

            res = len(re.findall(r"\w+", doc))

            if res > MAX_WORDS:
                st.warning(
                    "⚠️ Your text contains "
                    + str(res)
                    + " words."
                    + " Only the first 256 words will be reviewed. Stay tuned as increased allowance is coming! 😊"
                )

            if res == 0:
                with c2:
                    st.warning(
                        "⚠️ Please enter your comment before Analysing ")
                    st.stop()

            if ModelType == "None":
                st.warning(
                    "⚠️ Please Choose a Model for the prediction ")
                st.stop()

            doc = doc[:MAX_WORDS]
            
            with c2:

                if ModelType != "Ensemble Three Models":
                    
                    my_bar = st.progress(0)

                    processed_data = model.process(doc)

                    for percent_complete in range(81):
                        time.sleep(0.01)
                        my_bar.progress(percent_complete + 10)

                    preds = model.predict(processed_data)
                    label = preds.argmax(1)-1

                    my_bar.progress(percent_complete + 20) 

                    prob = max(preds[0])

                    if label == 0:
                        st.warning('That was Neutral Comment with **'+str(round(prob,4)*100)+'%** probability 😐')

                    elif label == 1:
                        st.success('That was Positive Comment **'+str(round(prob,4)*100)+'%** probability ✔️')

                    else :
                        st.error('That was Negative Comment **'+str(round(prob,4)*100)+'%** probability ❌')

                else:
                        
                    st.write("All models...") 
                    big_bar = st.progress(0)
                    all_probs = []
                    model_label = ["BERT Multilingual", "BERT Camembert (French)", "Arabic BERT"]
                    percent = 0 

                    for model,name in zip(list_model,model_label):

                        st.write(str(name) + " is Running...") 
                        my_bar = st.progress(0)
                        processed_data = model.process(doc)

                        for percent_complete in range(81):
                            time.sleep(0.01)
                            my_bar.progress(percent_complete + 10)

                        preds = model.predict(processed_data)
                        
                        my_bar.progress(percent_complete + 20) 

                        all_probs.append(preds)

                        label = preds.argmax(1)-1

                        prob = max(preds[0])
                    
                        if label == 0:
                            st.warning("**"+str(name)+'** Predicted That was Neutral Comment with **'+str(round(prob,4))+'%** probability 😐')

                        elif label == 1:
                            st.success("**"+str(name)+'** Predicted That was Positive Comment **'+str(round(prob,4))+'%** probability ✔️')

                        else :
                            st.error("**"+str(name)+'** Predicted That was Negative Comment **'+str(round(prob,4))+'%** probability ❌')
                    


                        big_bar.progress(percent + 33)
                        percent+=33

                    big_bar.progress(100)
                        
                    ensemble_prob = np.mean(all_probs, axis=0)
                    label = ensemble_prob.argmax(1)-1
                    prob = max(ensemble_prob[0])

                    st.markdown("""---""")
                    if label == 0:
                            st.warning('**Ensembling** Predicted That was Neutral Comment with **'+str(round(prob,4))+'%** probability 😐')

                    elif label == 1:
                        st.success('**Ensembling** Predicted That was Positive Comment **'+str(round(prob,4))+'%** probability ✔️')

                    else :
                        st.error('**Ensembling** Predicted That was Negative Comment **'+str(round(prob,4))+'%** probability ❌')



