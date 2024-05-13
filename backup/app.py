from flask import Flask, request, jsonify, url_for
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.gradient import GradientEmbedding
from llama_index.llms.gradient import GradientBaseModelLLM
from pathlib import Path
from wav2lip import inference 
from wav2lip.models import Wav2Lip
from gtts import gTTS
from io import BytesIO
import os
import torch
import shutil
from flask_cors import CORS,cross_origin



app = Flask(__name__)

CORS(app)


device = 'cpu'
# Load Wav2Lip model
def load_model(path):
    # Download model weights if not available
    wav2lip_checkpoints_url = "https://drive.google.com/drive/folders/1Sy5SHRmI3zgg2RJaOttNsN3iJS9VVkbg?usp=sharing"
    # if not os.path.exists(path):
    #     gdown.download_folder(wav2lip_checkpoints_url, quiet=True, use_cookies=False)
    
    # Load the model
    model = Wav2Lip()
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    model = model.to(device)
    return model.eval()

os.environ['GRADIENT_ACCESS_TOKEN'] = "vcHibVdzc50A18DsThmx0Mm0zc2TLeea"
os.environ['GRADIENT_WORKSPACE_ID'] = "bf886fbe-0ae1-4660-923d-d78bfe26f01f_workspace"

# Load Llama Index settings
Settings.llm = GradientBaseModelLLM(base_model_slug="llama2-7b-chat", max_tokens=400)
Settings.embed_model = GradientEmbedding(
    gradient_access_token=os.environ["GRADIENT_ACCESS_TOKEN"],
    gradient_workspace_id=os.environ["GRADIENT_WORKSPACE_ID"],
    gradient_model_slug="bge-large"
)

def copy_video_to_static(video_src, static_dst):
    shutil.copy(video_src, static_dst)

# Chat with PDF based on user's question
@app.route('/chat', methods=['POST'])
@cross_origin()
def chat_with_pdf():
    # Extract question from the JSON payload
    question = request.form['question']
    # question = payload['question']
    print("question:- ",question)
    # Load PDF documents
    documents = SimpleDirectoryReader("Documents").load_data()
    print("after")
    # Create VectorStoreIndex and query engine
    index = VectorStoreIndex.from_documents(documents, service_context=Settings)
    query_engine = index.as_query_engine()
    print("after1")

    # Perform chatbot logic based on the question
    pdf_response = query_engine.query(question)
    cleaned_response = pdf_response.response
    print("after2")

    # Generate response audio
    # sound_file = BytesIO()
    # tts = gTTS(cleaned_response, lang='en')
    # tts.write_to_fp(sound_file)
    # tts.save("welcome.wav")
    # sound_file.seek(0)
    # audio_bytes = sound_file.read()
    print("after3")

    # Generate video response
    # image_path = "avatars_images/avatar5.png"  # Placeholder for image path
    # textts = "welcome.wav"  # Placeholder for text to speech output
    print("after4")
    # model = load_model("wav2lip_checkpoints/wav2lip_gan.pth")
    print("after5")
    # inference.main(image_path, textts, model)
    print("cleaned_response:- ", cleaned_response)
    
    
    # Copy video to static folder
    # video_src = "wav2lip/results/result_voice.mp4"
    # static_dst = os.path.join("static", "result_voice.mp4")
    # copy_video_to_static(video_src, static_dst)

    # Construct video URL
    # video_url = url_for('static', filename='result_voice.mp4')
    
    
    # video_url = url_for('wav2lip/results', filename='result_voice.mp4')
    
    # Return response as JSON
    response = {
        'assistant': cleaned_response,
        # 'audio': audio_bytes,  # Send audio bytes
        # 'video_url': request.host + video_url  # Placeholder for video URL
    }
    # print("response",response)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
