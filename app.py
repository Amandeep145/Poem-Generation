from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st
import torch

st.set_page_config(
    page_title="Multipage App"
)
st.sidebar.success("Select a page")




base="dark"
primaryColor="purple"
st.title("Poem Generator")
st.markdown("This is a poem generator app. you can use it to create poems by providing text to it.")
@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = GPT2Tokenizer.from_pretrained("Silvers-145/khayal-generate")
    model = GPT2LMHeadModel.from_pretrained("Silvers-145/khayal-generate")
    return tokenizer, model

tokenizer,model = get_model()

# # load the model
# #model = GPT2LMHeadModel.from_pretrained("Silvers-145/khayal-generate")
# #tokenizer = GPT2Tokenizer.from_pretrained("Silvers-145/khayal-generate")

#model.eval()

#prompt = input("Enter the start of poem here:")
#button = st.button("Create")

prompt = st.text_area("Enter Text:")
click = st.button("Create")
if click and prompt:
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

    sample_outputs = model.generate(
        generated,
        do_sample=True,
        top_k=50,
        max_length=300,
        top_p=0.95,
        num_return_sequences=3
    )

    for i, sample_output in enumerate(sample_outputs):
        poemm = tokenizer.decode(sample_output, skip_special_tokens=True)
    # print("{}: {}\n\n".format(i,

    st.text_area("Generated Poem:", poemm)
    st.snow()
    st.success("Poem Generated")




