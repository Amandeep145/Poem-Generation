import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import streamlit as st


def add_bg_from_url():
  st.markdown(
    f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1578426187376-19bd5aeaeaa0?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1887&q=80");
             background-attachment: fixed;
             background-size: cover
         }}
         [data-testid="stHeader"] {{
         background-color: rgba(0,0,0,0);
         }}
         [data-testid="stSidebar"] {{
         background-image: url("https://images.unsplash.com/photo-1487528742387-d53d4f12488d?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1922&q=80");
         background-size: cover;
         }}
         [data-testisd="stVerticalBlock"]{{
         background-color: rgba(0,0,0,0);
         }}
         </style>

         """,
    unsafe_allow_html=True
  )


add_bg_from_url()



st.title("Text Paraphraser")
st.markdown("This is a Paraphraser app. you can use it to create paraphrases of the text by providing text to it.")

#tab1, tab2 = st.tabs(["Sentence Paraphraser", "Paragraph Paraphraser"])

#@st.cache(allow_output_mutation=True)
#def get_model_2():
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)  #return tokenizer,model,torch_device

#tokenizer,model = get_model_2()

#@st.cache(allow_output_mutation=True)
def get_response(input_text,num_return_sequences):
  batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

from sentence_splitter import SentenceSplitter, split_text_into_sentences


#choice = st.selectbox("Pick Paraphraser",["Sentence Paraphraser","Paragraph Paraphraser"])

#with tab1:
st.header("Sentence Paraphraser")
prompt2 = st.text_area("Enter Text:",height=200)
click2 = st.button("Create paraphrase sentence")
if prompt2 and click2:
    z = get_response(prompt2, 5)
    st.text_area("Paraphrased Sentence:", z,height=350)

#
# with tab2:
#    st.header("Paragraph Paraphraser")
#    st.text_area("Enter Paragraph:", height=200)
#    click3 = st.button("Create paraphrased paragraph")
#    if click3:
#      splitter = SentenceSplitter(language='en')
#      sentence_list = splitter.split(prompt2)
#      paraphrase = []
#      for i in sentence_list:
#        a = get_response(i, 1)
#        paraphrase.append(a)
#      paraphrase2 = [' '.join(x) for x in paraphrase]
#      paraphrase3 = [' '.join(x for x in paraphrase2)]
#      paraphrased_text = str(paraphrase3).strip('[]').strip("'")
#      st.text_area("Paraphrased Paragraph:", paraphrased_text,height=250)
#






#


#if prompt2 and click2:



# Paragraph of text
#context = input()
#print(context)



#print(paraphrased_text)