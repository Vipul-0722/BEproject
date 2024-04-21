from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tqdm import tqdm
from flask_cors import CORS

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# Constants
max_length = 40  # Max length of caption
attention_features_shape = 64  # Shape of attention features

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the saved encoder and decoder models
encoder = tf.keras.models.load_model('encoder.h5')
decoder = tf.keras.models.load_model('decoder.h5')

# Load VGG16 for image preprocessing
VGG16_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')

# Function to preprocess image
def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    return img

# Function to evaluate and generate caption
def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image), 0)
    img_tensor_val = VGG16_model.predict(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

@app.route('/after', methods=['POST'])
def after():
    file = request.files['file']
    file.save('static/file.jpg')

    # Generate caption
    caption, _ = evaluate('static/file.jpg')

    return jsonify({'caption': ' '.join(caption)})

if __name__ == "__main__":
    app.run(debug=True)
