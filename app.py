from flask import Flask, request, render_template, redirect, url_for, jsonify
from huggingface_hub import hf_hub_download
import os
import numpy as np
import cv2
import tensorflow as tf
import json

app = Flask(__name__)

# Download and load model
repo_id = "stephendsouza/Starlight-Tracker"
model_filename = "constellation_model.keras"
model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
model = tf.keras.models.load_model(model_path)
# Map predicted class index to constellation names with detailed information
constellation_details = {
    "Andromeda": {
        "description": "Andromeda is a constellation located in the northern sky, named after the mythological princess Andromeda.",
        "mythology": "In Greek mythology, Andromeda was chained to a rock as a sacrifice to a sea monster but was rescued by Perseus.",
        "facts": "The Andromeda Galaxy (M31), visible with the naked eye, is located within this constellation and is the nearest spiral galaxy to the Milky Way.",
        "best_visibility": "Autumn (September to November)",
        "significance": "It is home to the Andromeda Galaxy, the largest galaxy in the Local Group.",
        "view_more": "https://www.google.com/search?q=Andromeda+constellation"
    },
    "Aquila": {
        "description": "Aquila is a constellation in the northern sky, symbolizing an eagle.",
        "mythology": "Aquila represents Zeus's eagle, known for carrying the thunderbolts of the Greek god.",
        "facts": "Its brightest star, Altair, is part of the Summer Triangle asterism.",
        "best_visibility": "Summer (July to September)",
        "significance": "Known for its association with the Milky Way and the Summer Triangle.",
        "view_more": "https://www.google.com/search?q=Aquila+constellation"
    },
    "Auriga": {
        "description": "Auriga, the charioteer, is a bright constellation in the northern sky.",
        "mythology": "The constellation is linked to mythological figures such as Erichthonius, who invented the chariot.",
        "facts": "It contains Capella, the sixth-brightest star in the sky, and several star clusters.",
        "best_visibility": "Winter (December to February)",
        "significance": "Rich in star clusters like M36, M37, and M38, making it a favorite for stargazers.",
        "view_more": "https://www.google.com/search?q=Auriga+constellation"
    },
    "Canis Major": {
        "description": "Canis Major is a southern constellation representing the larger dog following Orion.",
        "mythology": "In Greek mythology, it represents one of Orion's hunting dogs.",
        "facts": "Sirius, the brightest star in the night sky, is located in this constellation.",
        "best_visibility": "Winter (December to February)",
        "significance": "Home to Sirius, also known as the Dog Star, a key navigation point in ancient times.",
        "view_more": "https://www.google.com/search?q=Canis+Major+constellation"
    },
    "Capricornus": {
        "description": "Capricornus, the sea-goat, is one of the zodiac constellations.",
        "mythology": "Associated with the goat-god Pan, who transformed into a sea-goat to escape Typhon.",
        "facts": "This faint constellation is home to the stars Algedi and Dabih, and the meteor shower Capricornids.",
        "best_visibility": "Late Summer to Early Fall (August to October)",
        "significance": "A zodiac constellation with historical ties to agriculture and the winter solstice.",
        "view_more": "https://www.google.com/search?q=Capricornus+constellation"
    },
    "Cetus": {
        "description": "Cetus is a constellation in the equatorial region, symbolizing a sea monster or whale.",
        "mythology": "In Greek mythology, it represents the sea monster sent to attack Andromeda.",
        "facts": "It contains Mira, a variable star whose brightness changes over a period of 332 days.",
        "best_visibility": "Autumn (October to December)",
        "significance": "Hosts Mira, one of the most famous variable stars.",
        "view_more": "https://www.google.com/search?q=Cetus+constellation"
    },
    "Gemini": {
        "description": "Gemini, the twins, is a prominent zodiac constellation.",
        "mythology": "It represents the twin brothers Castor and Pollux, known for their brotherly devotion.",
        "facts": "It is home to the bright stars Castor and Pollux, as well as the Geminid meteor shower.",
        "best_visibility": "Winter (December to February)",
        "significance": "Known for the Geminid meteor shower and its zodiac association.",
        "view_more": "https://www.google.com/search?q=Gemini+constellation"
    },
    "Leo": {
        "description": "Leo is a zodiac constellation representing a lion.",
        "mythology": "In Greek mythology, Leo represents the Nemean Lion defeated by Hercules.",
        "facts": "It contains Regulus, a bright star also known as the 'Heart of the Lion.'",
        "best_visibility": "Spring (March to May)",
        "significance": "A key zodiac constellation, easily recognizable for its lion-like shape.",
        "view_more": "https://www.google.com/search?q=Leo+constellation"
    },
    "Orion": {
        "description": "Orion is one of the most recognizable constellations in the night sky.",
        "mythology": "Named after Orion the Hunter, a figure in Greek mythology.",
        "facts": "It contains Betelgeuse and Rigel, along with the Orion Nebula, a stellar nursery.",
        "best_visibility": "Winter (December to February)",
        "significance": "Known for its iconic Belt stars and the Orion Nebula.",
        "view_more": "https://www.google.com/search?q=Orion+constellation"
    },
    "Sagittarius": {
        "description": "Sagittarius, the archer, is a zodiac constellation rich in deep-sky objects.",
        "mythology": "It represents a centaur archer, associated with the Greek myth of Chiron.",
        "facts": "It contains the Galactic Center and many star clusters and nebulae.",
        "best_visibility": "Summer (July to September)",
        "significance": "Home to the Galactic Center and notable nebulae like the Lagoon Nebula.",
        "view_more": "https://www.google.com/search?q=Sagittarius+constellation"
    },
    "Taurus": {
        "description": "Taurus, the bull, is a prominent zodiac constellation.",
        "mythology": "Associated with Zeus, who transformed into a bull to abduct Europa.",
        "facts": "It contains the Pleiades and Hyades star clusters and the bright star Aldebaran.",
        "best_visibility": "Winter (December to February)",
        "significance": "Known for its bright star Aldebaran and the Pleiades star cluster.",
        "view_more": "https://www.google.com/search?q=Taurus+constellation"
    },
    "Scorpius": {
        "description": "Scorpius is a zodiac constellation representing a scorpion.",
        "mythology": "In Greek mythology, it represents the scorpion sent to kill Orion.",
        "facts": "It contains Antares, a bright red supergiant star, and several star clusters.",
        "best_visibility": "Summer (July to September)",
        "significance": "Known for its bright stars and dramatic appearance.",
        "view_more": "https://www.google.com/search?q=Scorpius+constellation"
    },
    "Libra": {
        "description": "Libra, the scales, is a zodiac constellation.",
        "mythology": "It symbolizes balance and is associated with the goddess of justice, Themis.",
        "facts": "It contains the stars Zubenelgenubi and Zubeneschamali.",
        "best_visibility": "Spring (April to June)",
        "significance": "Known for its association with justice and balance.",
        "view_more": "https://www.google.com/search?q=Libra+constellation"
    },
    "Pisces": {
        "description": "Pisces is a zodiac constellation representing two fish tied together by a cord.",
        "mythology": "In Greek mythology, it represents the fish into which Aphrodite and her son Eros transformed to escape Typhon.",
        "facts": "Although not bright, it is significant as the location of the vernal equinox in the Age of Pisces.",
        "best_visibility": "Autumn (October to December)",
        "significance": "Zodiac constellation symbolizing the fish and associated with water.",
        "view_more": "https://www.google.com/search?q=Pisces+constellation"
    },
    "Virgo": {
        "description": "Virgo is the second-largest constellation and a zodiac constellation representing a maiden.",
        "mythology": "Linked to the goddess Demeter and the story of Persephone's abduction.",
        "facts": "It contains Spica, a bright binary star, and several galaxies like the Sombrero Galaxy.",
        "best_visibility": "Spring (April to June)",
        "significance": "Known for its bright star Spica and as a rich galaxy field.",
        "view_more": "https://www.google.com/search?q=Virgo+constellation"
    },
    "Ursa Major": {
        "description": "Ursa Major, the Great Bear, is one of the largest and most recognizable constellations in the northern sky.",
        "mythology": "In Greek mythology, it represents Callisto, who was turned into a bear by Hera.",
        "facts": "Home to the Big Dipper asterism, which is part of Ursa Major.",
        "best_visibility": "Spring (March to May)",
        "significance": "The Big Dipper serves as a pointer to Polaris, the North Star.",
        "view_more": "https://www.google.com/search?q=Ursa+Major+constellation"
    },
    "Ursa Minor": {
        "description": "Ursa Minor, the Little Bear, contains the North Star, Polaris.",
        "mythology": "It represents Arcas, son of Callisto, in Greek mythology.",
        "facts": "Polaris is part of the Little Dipper, an asterism within Ursa Minor.",
        "best_visibility": "Year-round in the Northern Hemisphere",
        "significance": "Home to Polaris, an important star for navigation.",
        "view_more": "https://www.google.com/search?q=Ursa+Minor+constellation"
    },
    "Lyra": {
        "description": "Lyra is a small constellation representing a lyre or harp.",
        "mythology": "Linked to Orpheus, the Greek musician and poet who played a lyre.",
        "facts": "It contains Vega, the fifth-brightest star in the sky, and the Ring Nebula (M57).",
        "best_visibility": "Summer (July to September)",
        "significance": "Known for its bright star Vega and the Ring Nebula.",
        "view_more": "https://www.google.com/search?q=Lyra+constellation"
    },
    "Cassiopeia": {
        "description": "Cassiopeia is a W-shaped constellation in the northern sky.",
        "mythology": "Named after Queen Cassiopeia, who boasted about her beauty in Greek mythology.",
        "facts": "It is easily recognizable for its W or M shape and contains the Cassiopeia A supernova remnant.",
        "best_visibility": "Autumn (September to November)",
        "significance": "Used in navigation due to its distinctive shape near the North Star.",
        "view_more": "https://www.google.com/search?q=Cassiopeia+constellation"
    },
    "Perseus": {
        "description": "Perseus is a northern constellation named after the Greek hero Perseus.",
        "mythology": "Represents the hero who slew Medusa and saved Andromeda.",
        "facts": "It contains the famous variable star Algol and the Perseid meteor shower.",
        "best_visibility": "Autumn (September to November)",
        "significance": "Home to the famous Algol star, also called the Demon Star.",
        "view_more": "https://www.google.com/search?q=Perseus+constellation"
    },
    "Draco": {
        "description": "Draco, the dragon, is a northern circumpolar constellation.",
        "mythology": "In Greek mythology, it represents the dragon slain by Hercules.",
        "facts": "It contains Thuban, a former North Star, and the Cat's Eye Nebula.",
        "best_visibility": "Year-round in the Northern Hemisphere",
        "significance": "Known for its historical significance and unique shape.",
        "view_more": "https://www.google.com/search?q=Draco+constellation"
    }
}


# Function to extract star-like points (same as before)
def extract_star_coordinates(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (300, 300))
    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points.append((cx, cy))
    normalized_points = [{"x": (x / 150) - 1, "y": 1 - (y / 150), "z": 0} for x, y in points]
    return normalized_points[:20]

# Function to predict constellation (same as before)
def predict_constellation(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return list(constellation_details.keys())[predicted_class]

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Loading page
@app.route('/loading/<image>')
def loading(image):
    img_path = os.path.join('static', 'uploads', image)
    star_coordinates = extract_star_coordinates(img_path)
    return render_template('loading.html', image=image, stars=json.dumps(star_coordinates))

# Upload handler
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' in request.files:
        img = request.files['image']
        uploads_dir = os.path.join('static', 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        img_path = os.path.join(uploads_dir, img.filename)
        img.save(img_path)
        return redirect(url_for('loading', image=img.filename))
    return redirect(url_for('home'))
# Result page
@app.route('/result/<image>')
def result(image):
    # Construct the image path based on the uploaded image
    img_path = os.path.join('static', 'uploads', image)
    
    # Predict the constellation based on the image
    predicted_constellation = predict_constellation(img_path)
    
    # Get the information for the predicted constellation
    constellation_info = constellation_details.get(predicted_constellation, {})
    
    # Pass the necessary information to the template
    return render_template('result.html', 
                           constellation=predicted_constellation, 
                           image=image, 
                           description=constellation_info.get("description", "No description available"), 
                           mythology=constellation_info.get("mythology", "No mythology available"),
                           facts=constellation_info.get("facts", "No facts available"),
                           best_visibility=constellation_info.get("best_visibility", "Visibility data not available"),
                           significance=constellation_info.get("significance", "Significance data not available"),
                           view_more=constellation_info.get("view_more", "#"))\
# Run the Flask app
# Run the Flask app
if __name__ == '__main__':
    # Ensure the uploads directory exists
    os.makedirs('static/uploads', exist_ok=True)
    
    # Get the port from the environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app on the correct port and make it accessible externally
    app.run(debug=True, host='0.0.0.0', port=port)
