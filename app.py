import streamlit as st
import torch
import cv2
import tempfile
import numpy as np

# from src.models import VideoVAE, MLPClassifier
# from src.dataset import preprocess_frames
# from src.config import cfg

st.title("Visual Speech Recognition (Lipreading)")
st.write("Upload a video of someone speaking (GRID format), and the model will attempt to read their lips!")

uploaded_file = st.file_uploader("Choose a video file", type=["mpg", "mp4", "avi"])

if uploaded_file is not None:
    # Save uploaded file to temp file to read with OpenCV
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    st.video(tfile.name)
    
    if st.button("Predict Word"):
        with st.spinner("Processing video..."):
            # Provide an industrial mock or load actual model here based on 'MLPClassifier_full.pt' and 'vae_trained.pt'
            
            # Example Pipeline structure you would implement:
            from src.models import VideoVAE, MLPClassifier
            from src.dataset import preprocess_frames, extract_word_frames
            from src.config import cfg
            import torch
            import cv2
            
            # 1. Load VAE: 
            # Set use_pretrained=False so it doesn't try to download base weights we are just going to overwrite
            vae = VideoVAE(latent_dim=cfg.LATENT_DIM, num_classes=cfg.NUM_CLASSES, use_pretrained=False).to(cfg.DEVICE)
            vae.load_state_dict(torch.load('data/vae_trained.pt', map_location=cfg.DEVICE))
            vae.eval()
            
            # 2. Extract frames via cv2 and preprocess_frames()
            # For inference without an .align file, we process the whole video
            cap = cv2.VideoCapture(tfile.name)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            
            video_tensor = preprocess_frames(frames).to(cfg.DEVICE)
            
            # 3. get VAE encoding (mu, logvar)
            with torch.no_grad():
                mu, logvar = vae.encode(video_tensor)
                
                # The MLP expects concatenated logvar and mu if USE_MU_ONLY is false
                features = torch.cat((logvar, mu), dim=1)
            
            # 4. Pass thru MLP
            # Use weights_only=False because the model was saved as an entire object, not just a state_dict
            mlp = torch.load('data/MLPClassifier_full.pt', map_location=cfg.DEVICE, weights_only=False)
            mlp.eval()
            
            with torch.no_grad():
                logits = mlp(features)
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
                
                # Load the original training labels to recover the vocabulary mapping
                try:
                    train_labels = torch.load('data/tr_labels.pt')
                    unique_labels = sorted(set(train_labels))
                    
                    # Optional: Force the model to ignore the "sil" (silence) class 
                    # since full uncropped videos contain a lot of silence.
                    if "sil" in unique_labels:
                        sil_idx = unique_labels.index("sil")
                        probs[sil_idx] = 0.0 # Zero out the silence probability
                        
                except Exception as e:
                    unique_labels = None
                    print(f"Warning: Could not load label dictionary: {e}")

                top_probs, top_idxs = torch.topk(probs, 3)
            
            if unique_labels:
                st.success(f"Top Prediction: **{unique_labels[top_idxs[0].item()]}** ({top_probs[0].item()*100:.1f}%)")
                st.write("Other likely words detected in the video:")
                st.write(f"- **{unique_labels[top_idxs[1].item()]}** ({top_probs[1].item()*100:.1f}%)")
                st.write(f"- **{unique_labels[top_idxs[2].item()]}** ({top_probs[2].item()*100:.1f}%)")
            else:
                st.success(f"Predicted Class Index: {top_idxs[0].item()}")
