from flask import Flask,render_template,request, jsonify
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50


# define ResNet50 model
# ResNet50_model_ = ResNet50(weights='imagenet')


from tensorflow.keras.applications.resnet50 import preprocess_input

breed_labels={ 151: 'Chihuahua',
 152: 'Japanese spaniel',
 153: 'Maltese dog, Maltese terrier, Maltese',
 154: 'Pekinese, Pekingese, Peke',
 155: 'Shih-Tzu',
 156: 'Blenheim spaniel',
 157: 'papillon',
 158: 'toy terrier',
 159: 'Rhodesian ridgeback',
 160: 'Afghan hound, Afghan',
 161: 'basset, basset hound',
 162: 'beagle',
 163: 'bloodhound, sleuthhound',
 164: 'bluetick',
 165: 'black-and-tan coonhound',
 166: 'Walker hound, Walker foxhound',
 167: 'English foxhound',
 168: 'redbone',
 169: 'borzoi, Russian wolfhound',
 170: 'Irish wolfhound',
 171: 'Italian greyhound',
 172: 'whippet',
 173: 'Ibizan hound, Ibizan Podenco',
 174: 'Norwegian elkhound, elkhound',
 175: 'otterhound, otter hound',
 176: 'Saluki, gazelle hound',
 177: 'Scottish deerhound, deerhound',
 178: 'Weimaraner',
 179: 'Staffordshire bullterrier, Staffordshire bull terrier',
 180: 'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier',
 181: 'Bedlington terrier',
 182: 'Border terrier',
 183: 'Kerry blue terrier',
 184: 'Irish terrier',
 185: 'Norfolk terrier',
 186: 'Norwich terrier',
 187: 'Yorkshire terrier',
 188: 'wire-haired fox terrier',
 189: 'Lakeland terrier',
 190: 'Sealyham terrier, Sealyham',
 191: 'Airedale, Airedale terrier',
 192: 'cairn, cairn terrier',
 193: 'Australian terrier',
 194: 'Dandie Dinmont, Dandie Dinmont terrier',
 195: 'Boston bull, Boston terrier',
 196: 'miniature schnauzer',
 197: 'giant schnauzer',
 198: 'standard schnauzer',
 199: 'Scotch terrier, Scottish terrier, Scottie',
 200: 'Tibetan terrier, chrysanthemum dog',
 201: 'silky terrier, Sydney silky',
 202: 'soft-coated wheaten terrier',
 203: 'West Highland white terrier',
 204: 'Lhasa, Lhasa apso',
 205: 'flat-coated retriever',
 206: 'curly-coated retriever',
 207: 'golden retriever',
 208: 'Labrador retriever',
 209: 'Chesapeake Bay retriever',
 210: 'German short-haired pointer',
 211: 'vizsla, Hungarian pointer',
 212: 'English setter',
 213: 'Irish setter, red setter',
 214: 'Gordon setter',
 215: 'Brittany spaniel',
 216: 'clumber, clumber spaniel',
 217: 'English springer, English springer spaniel',
 218: 'Welsh springer spaniel',
 219: 'cocker spaniel, English cocker spaniel, cocker',
 220: 'Sussex spaniel',
 221: 'Irish water spaniel',
 222: 'kuvasz',
 223: 'schipperke',
 224: 'groenendael',
 225: 'malinois',
 226: 'briard',
 227: 'kelpie',
 228: 'komondor',
 229: 'Old English sheepdog, bobtail',
 230: 'Shetland sheepdog, Shetland sheep dog, Shetland',
 231: 'collie',
 232: 'Border collie',
 233: 'Bouvier des Flandres, Bouviers des Flandres',
 234: 'Rottweiler',
 235: 'German shepherd, German shepherd dog, German police dog, alsatian',
 236: 'Doberman, Doberman pinscher',
 237: 'miniature pinscher',
 238: 'Greater Swiss Mountain dog',
 239: 'Bernese mountain dog',
 240: 'Appenzeller',
 241: 'EntleBucher',
 242: 'boxer',
 243: 'bull mastiff',
 244: 'Tibetan mastiff',
 245: 'French bulldog',
 246: 'Great Dane',
 247: 'Saint Bernard, St Bernard',
 248: 'Eskimo dog, husky',
 249: 'malamute, malemute, Alaskan malamute',
 250: 'Siberian husky',
 251: 'dalmatian, coach dog, carriage dog',
 252: 'affenpinscher, monkey pinscher, monkey dog',
 253: 'basenji',
 254: 'pug, pug-dog',
 255: 'Leonberg',
 256: 'Newfoundland, Newfoundland dog',
 257: 'Great Pyrenees',
 258: 'Samoyed, Samoyede',
 259: 'Pomeranian',
 260: 'chow, chow chow',
 261: 'keeshond',
 262: 'Brabancon griffon',
 263: 'Pembroke, Pembroke Welsh corgi',
 264: 'Cardigan, Cardigan Welsh corgi',
 265: 'toy poodle',
 266: 'miniature poodle',
 267: 'standard poodle',
 268: 'Mexican hairless',}
emotion_labels = ['Angry', 'fear', 'Happy', 'Neutral', 'Sad']

def load_model(model_path):
  """
  Loads a saved model from a specified path.
  """
  # print(f"Loading saved model from: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
  return model

breedModel=ResNet50(weights='imagenet')
emotionModel=load_model('assets/models/mobilenet_v2_emotion_new_sgd_fine_neww_1.h5')

app=Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict', methods=[ 'POST', 'GET'])
def predict():
    data = {}
    if request.method == 'POST':

        img = request.files['select_file']

        img.save('static/pic.jpg')

        image_path = "static/pic.jpg"

        data=predictImage(image_path)

    return jsonify(data),200


def predictImage(image_path):
    # Emotion classification
    # coverting to tenosors
    data = tf.constant(image_path)
    # Read in image file
    image = tf.io.read_file(data)
    # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
    image = tf.image.decode_jpeg(image, channels=3)
    # Convert the colour channel values from 0-225 values to 0-1 values
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize the image to our desired size (224, 244)
    image = tf.image.resize(image, size=[224, 224])

    # reshaping to input size of the moedel
    image = tf.reshape(image, [1, 224, 224, 3])

    emotion_preds = emotionModel.predict(image)
    data = emotion_labels[np.argmax(emotion_preds)]
    percentage = np.max(emotion_preds) * 100

    #
    top_emo = emotion_preds.argsort()[0][::-1][:4]  # taking top 4 predictions indexes
    second_emo = emotion_labels[top_emo[1]]  # second prediction
    second_emo_per = emotion_preds[0][top_emo[1]] * 100
    third_emo = emotion_labels[top_emo[2]]  # third
    third_emo_per = emotion_preds[0][top_emo[2]] * 100

    # breed classification
    # loads RGB image as PIL.Image.Image type
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = tf.keras.preprocessing.image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    img = np.expand_dims(x, axis=0)
    img = preprocess_input(img)
    breed_preds = breedModel.predict(img)

    num = np.argmax(breed_preds)

    if (num <= 268) & (num >= 151):
        breed = breed_labels[num]

    else:
        breed = 'Not a Dog Image or Image Non Classifiable'
    # preds=model.predict(image)
    # lbl = np.argmax(preds)
    # data = class_labels[lbl]
    data= {'breed':breed,
                    'emotion':data,
                    'percentage':percentage,
                    "second_label":{'second_emotion':second_emo,
                                     'second_emotion_percent':second_emo_per},
                    "third_label": {'third_emotion':third_emo,
                                    'third_emotion_percent':third_emo_per}}
    return data

if __name__=="__main__":
    app.run(debug=False,host='127.0.0.1',port=8080)
