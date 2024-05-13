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

device='cpu'
@st.cache_resource
def create_datastax_connection():

    cloud_config= {'secure_connect_bundle': 'secure-connect-temp-db.zip'}
    with open("bharat_astra_token.json") as f:
        secrets = json.load(f)

    CLIENT_ID = secrets["clientId"]
    CLIENT_SECRET = secrets["secret"]

    auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    astra_session = cluster.connect()
    return astra_session

@st.cache_resource
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



image_video_map = {
      				# "avatars_images/avatar1.jpg":"avatars_videos/avatar1.mp4",
                    "avatars_images/avatar2.jpg":"avatars_videos/avatar2.mp4",
                    # "avatars_images/avatar3.png":"avatars_videos/avatar3.mp4"
                              }


def streamlit_look():
    """
    Modest front-end code:)
    """
    data={}
    
    st.write("Please choose your avatar from the following options:")
    # avatar_img = image_select("", [
    #                             # "avatars_images/avatar1.jpg",
    #       						"avatars_images/avatar2.jpg",
    #                             # "avatars_images/avatar3.png",
    #                                     ])
    avatar_img = "avatars_images/avatar5.png"
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


def main():
    docs_path = "Documents"
    index_placeholder = None
    st.set_page_config(page_title = "Chat with your PDF using Llama2 & Llama Index", page_icon="🦙")
    st.header('Chat with your PDF')
    
    # print("session_state-=-=-=-=-=-=-=",st.session_state)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar = message['avatar']):
            st.markdown(message["content"])

    # session = create_datastax_connection()
    # print("session-=-=-=-=-",session)

    os.environ['GRADIENT_ACCESS_TOKEN'] = "vcHibVdzc50A18DsThmx0Mm0zc2TLeea"
    os.environ['GRADIENT_WORKSPACE_ID'] = "bf886fbe-0ae1-4660-923d-d78bfe26f01f_workspace"

    # llm = GradientBaseModelLLM(base_model_slug="llama2-7b-chat", max_tokens=400)

    # embed_model = GradientEmbedding(
    #     gradient_access_token = os.environ["GRADIENT_ACCESS_TOKEN"],
    #     gradient_workspace_id = os.environ["GRADIENT_WORKSPACE_ID"],
    #       gradient_model_slug="bge-large")

    # service_context = ServiceContext.from_defaults(
    # llm = llm,
    # embed_model = embed_model,
    # chunk_size=256)

    # set_global_service_context(service_context)
    
    Settings.llm = GradientBaseModelLLM(base_model_slug="llama2-7b-chat", max_tokens=400)
    Settings.embed_model = GradientEmbedding(
        gradient_access_token = os.environ["GRADIENT_ACCESS_TOKEN"],
        gradient_workspace_id = os.environ["GRADIENT_WORKSPACE_ID"],
        gradient_model_slug="bge-large")

    with st.sidebar:
        st.subheader('Upload Your PDF File')
        docs = st.file_uploader('⬆️ Upload your PDF & Click to process',
                                accept_multiple_files = False, 
                                type=['pdf','docs','docx'])
        # print("docs",docs)
        if st.button('Process'):
            save_path = Path(docs_path,docs.name)
            with open(save_path,"wb") as w:
                w.write(docs.getvalue())
            if save_path.exists():
                st.success(f'File {docs.name} is successfully saved!')
            with NamedTemporaryFile(dir='.', suffix='.pdf') as f:
                f.write(docs.getbuffer())
                print("NamedTemporaryFile-=-=-=-=--==-=-=")
                # print("f-=-=-=-=-=-=-=-",f)
                with st.spinner('Processing'):
                    documents = SimpleDirectoryReader("Documents").load_data()
                    # print("documents-=-=-=-=-=-=-=-=-",documents)
                    index = VectorStoreIndex.from_documents(documents,
                                                            service_context=Settings)
                    print("index-=-=-=-=--==-=-=")
                    query_engine = index.as_query_engine()
                    if "query_engine" not in st.session_state:
                        print("query_engine-=-=-=-=--==-=-=")
                        st.session_state.query_engine = query_engine
                    st.session_state.activate_chat = True

    print("st.session_state.activate_chat-=-=-=-=",st.session_state.activate_chat)
    if st.session_state.activate_chat == True:
        print("st.session_state-=-=-=-=--=--=-",st.session_state.activate_chat)
        if prompt := st.chat_input("Ask your question from the PDF?"):
            print("inside prompt st.session_state-=-=-=-=--=--=-",st.session_state.activate_chat)
            with st.chat_message("user", avatar = '👨🏻'):
                st.markdown(prompt)
                print("prompt")
            st.session_state.messages.append({"role": "user", 
                                              "avatar" :'👨🏻',
                                              "content": prompt})

            query_index_placeholder = st.session_state.query_engine
            pdf_response = query_index_placeholder.query(prompt)
            cleaned_response = pdf_response.response
            with st.chat_message("assistant", avatar='🤖'):
                print("cleaned_response")
                st.markdown(cleaned_response)
            st.session_state.messages.append({"role": "assistant", 
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
            if os.path.exists('wav2lip/results/result_voice.mp4'):
                st.video('wav2lip/results/result_voice.mp4')
                # video_url = 'wav2lip/results/result_voice.mp4'
                
                # st.markdown(f'<video autoplay controls><source src={video_url} type="video/mp4"></video><br/>wav2lip/results/result_voice.mp4'
                #             , unsafe_allow_html=True)

        else:
            st.markdown(
                'Upload your PDFs to chat'
                )


if __name__ == '__main__':
    main()
