import argparse
import time

import numpy as np
import streamlit as st
import torch
from transformers import RobertaTokenizer

import session_state  # streamlit patch to maintain state between choices
from run_offenseval import RobertaForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument("--modela-path", help="Path to model and tokenizer", required=True)
parser.add_argument("--modelb-path", help="Path to model and tokenizer", required=True)
parser.add_argument("--modelc-path", help="Path to model and tokenizer", required=True)
# parser.add_argument("--labels-path", help="Path to labels file", required=True)
args = parser.parse_args()
modela_path = args.modela_path
modelb_path = args.modelb_path
modelc_path = args.modelb_path
# labels_path = args.labels_path


class Root:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained(
            modela_path, do_lower_case=False
        )
        self.modela = RobertaForSequenceClassification.from_pretrained(modela_path)
        self.modelb = RobertaForSequenceClassification.from_pretrained(modelb_path)
        self.modelc = RobertaForSequenceClassification.from_pretrained(modelc_path)

    def classification(self, text, task="task_a"):
        sentences = [text]

        mask_padding_with_zero = True
        inputs = self.tokenizer.encode_plus(
            sentences[0],
            add_special_tokens=True,
            max_length=128,
        )
        input_ids = inputs["input_ids"]
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        input_ids = torch.tensor([input_ids], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long)

        with torch.no_grad():
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if task == "task_a":
                outputs = self.modela(**inputs)
            elif task == "task_b":
                outputs = self.modelb(**inputs)
            elif task == "task_c":
                outputs = self.modelc(**inputs)

        preds = outputs[0].numpy()

        preds = np.argmax(preds, axis=1)
        preds = preds.tolist()
        label = "UnKnown"
        if preds[0] == 0:
            if task == "task_b":
                label = "Targeted Insult"
            elif task == "task_c":
                label = "Individual"
            else:
                label = "Not Offensive" + "\n"
        elif preds[0] == 1:
            if task == "task_b":
                label = "Untargeted"
            elif task == "task_c":
                label = "Group"
            else:
                label = "Offensive"
        else:
            label = "Other"
        return label


@st.cache(allow_output_mutation=True)
def return_root_object():
    root = Root()
    return root


root_class = return_root_object()

s = session_state.get(
    bt_pressed_first_button=False,
    bt_recent_altered_text="",
    bt_recent_altered_area_text="",
    bt_recent_altered_select_text="",
)


st.header("Offensive Text Classification")
st.markdown(body="---")
task = st.radio(  "Select task",   ('task_a', 'task_b', 'task_c'))
dropdown_text=""
if task == "task_a":
    dropdown_text = st.selectbox(
        "Select text",
        options=["@USER go to hell", "@USER how are you",
        "@USER Why are you telling them to not do this? I encourage Antifa shits to try this and get dropped. Less of them the better."],
    )
elif task == "task_b":
    dropdown_text = st.selectbox(
        "Select text",
        options=["@USER @USER @USER @USER F*** yeah!!!", "@USER F*** you!",
        "@USER @USER @USER i think i missed a few replies here (im on a conference call) but most of the anti jihadi muslim vets i know would disagree with gun control. so this seems odd...?",
        "@USER I feel pretty fucked then cuz I started out fine"],
    )
elif task == "task_c":
    dropdown_text = st.selectbox(
        "Select text",
        options=["@USER @USER @USER @USER LOL!!!   Throwing the BULLSHIT Flag on such nonsense!!  #PutUpOrShutUp   #Kavanaugh   #MAGA   #CallTheVoteAlready URL",
         "@User F*** this society",
         "@USER Hopefully your obesity catches up with you by then"],
    )
TEXT = st.text_area(label="Enter Text to Classify")

button_clicked = st.button(label="Classify")


# if DE_TEXT and de_button_clicked:
#     output_text = root_class.remove_punctuation(DE_TEXT)
#     st.write("Copy to Clipboard the below text to test using it.")
#     st.markdown(body=f"```text\n{output_text}\n```")

if dropdown_text != s.bt_recent_altered_select_text:
    s.bt_recent_altered_select_text = dropdown_text
    s.bt_recent_altered_text = dropdown_text
if TEXT not in ("", s.bt_recent_altered_area_text):
    s.bt_recent_altered_area_text = TEXT
    s.bt_recent_altered_text = TEXT
if s.bt_recent_altered_text != "" and button_clicked:
    s.bt_pressed_first_button = True
    timer = time.time()
    response = root_class.classification(s.bt_recent_altered_text, task)
    # se = StringEdit(s.bt_recent_altered_text, response)
    # html = se.generate_html()
    # st.markdown(body='*Diff* :')
    # components.html(html, scrolling=True, height = 80) # import streamlit.components.v1 as components # pylint: disable=line-too-long
    s.bt_output = (
        response
        + f"  \n ##### Time-Taken : {(time.time()-timer)*1000:.2f} ms  Model : Roberta"
    )
if s.bt_pressed_first_button:
    st.markdown(body=s.bt_output)

# to hide hamburger symbol (top right corner) and made with streamlit footer (screencast, clear cache will not be available) # pylint: disable=line-too-long
HIDE_STREAMLIT_STYLE = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(body=HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)