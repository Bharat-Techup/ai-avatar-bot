import json
import os
import streamlit as st
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
# from llama_index.core import ServiceContext
from llama_index.core import Settings
# from llama_index.core import set_global_service_context
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
# from llama_index.embeddings.gradient  import GradientEmbedding
from llama_index.embeddings.gradient import GradientEmbedding
from llama_index.llms.gradient import GradientBaseModelLLM
# from llama_index.vector_stores import CassandraVectorStore
# from copy import deepcopy
from tempfile import NamedTemporaryFile
from pathlib import Path
from streamlit_image_select import image_select
import torch
from streamlit_mic_recorder import mic_recorder
from wav2lip import inference 
from wav2lip.models import Wav2Lip
import gdown
# from transformers import pipeline
from gtts import gTTS
from io import BytesIO

from ctransformers import AutoModelForCausalLM
from flask import Flask ,session ,request ,jsonify
from flask_caching import Cache



# Define a function to be cached


app = Flask(__name__)

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

device = 'cpu'
@app.route('/health_check', methods=['GET'])
def health_check():
    return "Project working fine"


@cache.cached(timeout=300)
def load_model(path):
    # st.write("Please wait for the model to be loaded or it will cause an error")
    wav2lip_checkpoints_url = "https://drive.google.com/drive/folders/1Sy5SHRmI3zgg2RJaOttNsN3iJS9VVkbg?usp=sharing"
    if not os.path.exists(path):
        gdown.download_folder(wav2lip_checkpoints_url, quiet=True, use_cookies=False)
    # st.write("Please wait")
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path,map_location=lambda storage, loc: storage)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    model = model.to(device)
    # st.write("model is loaded!")
    return model.eval()


def streamlit_look():
    """
    Modest front-end code:)
    """
    data={}
    

    avatar_img = "avatars_images/avatar2.jpg"
    data["imge_path"] = avatar_img
    audio=mic_recorder(
    # start_prompt="Start recording",
    # stop_prompt="Stop recording", 
    just_once=False,
    use_container_width=False,
    callback=None,
    args=(),
    kwargs={},
    key=None)
    if audio:
          st.audio(audio["bytes"])
          data["audio"]= audio["bytes"]
    return data


@app.route('/chat',methods=['POST'])
def main():
    if "conversation" not in session:
        session.conversation = None

    if "activate_chat" not in session:
        session.activate_chat = False

    if "messages" not in session:
        session.messages = []
       
    # messages = [
    # {"role": "user", "avatar": "👨🏻", "content": "Hello!"},
    # {"role": "assistant", "avatar": "🤖", "content": "Hi there!"},
    # # Add more messages as needed
    # ]  
        
    for message in session.messages:
        with session(message["role"], avatar = message['avatar']):
            return jsonify({"chat_bot":session.messages.message[-1] })
            
    print("hello")       
    os.environ['GRADIENT_ACCESS_TOKEN'] = "vcHibVdzc50A18DsThmx0Mm0zc2TLeea"
    os.environ['GRADIENT_WORKSPACE_ID'] = "bf886fbe-0ae1-4660-923d-d78bfe26f01f_workspace"
    
    Settings.llm = GradientBaseModelLLM(base_model_slug="llama2-7b-chat", max_tokens=400)
    Settings.embed_model = GradientEmbedding(
        gradient_access_token = os.environ["GRADIENT_ACCESS_TOKEN"],
        gradient_workspace_id = os.environ["GRADIENT_WORKSPACE_ID"],
        gradient_model_slug="bge-large")
    
    
    # with NamedTemporaryFile(dir='.', suffix='.pdf') as f:
    #     f.write(docs.getbuffer())
    documents = SimpleDirectoryReader("Documents").load_data()
    # print("documents-=-=-=-=-=-=-=-=-",documents)
    index = VectorStoreIndex.from_documents(documents,
                                            service_context=Settings)
    print("index-=-=-=-=--==-=-=")
    query_engine = index.as_query_engine()
    if "query_engine" not in session:
        print("query_engine-=-=-=-=--==-=-=")
        session.query_engine = query_engine
    session.activate_chat = True
    
    if session.activate_chat == True:
        if prompt := request.form['human']:
            with session.chat_message("user", avatar = '👨🏻'):
                session.markdown(prompt)
                print("prompt")
            session.messages.append({"role": "user", 
                                              "avatar" :'👨🏻',
                                              "content": prompt})
            
            
            query_index_placeholder = session.query_engine
            pdf_response = query_index_placeholder.query(prompt)
            cleaned_response = pdf_response.response
            with session.chat_message("assistant", avatar='🤖'):
                print("cleaned_response")
                st.markdown(cleaned_response)
            session.messages.append({"role": "assistant", 
                                              "avatar" :'🤖',
                                              "content": cleaned_response})
            
            sound_file = BytesIO()
            tts = gTTS(cleaned_response, lang='en')
            tts.write_to_fp(sound_file)
            tts.save("welcome.wav") 
            # st.audio(sound_file)
            textts = "welcome.wav"
            
            data=streamlit_look()
            model = load_model("wav2lip_checkpoints/wav2lip_gan.pth")
            # fast_animate = st.button("fast animate")
            
            # if fast_animate:
            inference.main(data["imge_path"],textts,model)
            
            return jsonify({
                "chat_bot":session.messages[-1] 
            })
            
    else:
        return jsonify("chat not activated")
    
    
if __name__ == '__main__':
    app.run(debug=True)