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
model_filename = "constellation_model_mobilenet.keras"
model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
model = tf.keras.models.load_model(model_path)
# Map predicted class index to constellation names with detailed information
constellation_details = {
    "Andromeda": {
        "description": "Andromeda is a prominent constellation located in the northern sky, named after the mythological princess Andromeda. It is one of the most recognizable constellations due to its significant position near the celestial equator and its proximity to the Milky Way.",
        "mythology": "In Greek mythology, Andromeda was a beautiful princess, daughter of King Cepheus and Queen Cassiopeia, who was chained to a rock as a sacrifice to appease the sea god Poseidon. She was saved by Perseus, who later married her. The constellation represents her in the sky, with the chain symbolizing her sacrifice.",
        "facts": "The Andromeda Galaxy (M31), which is visible with the naked eye, lies within this constellation. It is the nearest spiral galaxy to the Milky Way and is on a collision course with our galaxy, expected to merge in about 4.5 billion years. The Andromeda constellation is home to several other deep-sky objects, such as the Andromeda Nebula (M33), and hosts a variety of star clusters and galaxies.",
        "best_visibility": "Autumn (September to November), when the constellation is visible in the evening hours in the northern hemisphere. The best time for observing is typically during clear, dark nights in these months.",
        "significance": "Andromeda is home to the Andromeda Galaxy, the largest galaxy in the Local Group of galaxies, which includes the Milky Way, the Triangulum Galaxy, and about 30 other smaller galaxies. This makes it an important astronomical feature for understanding the structure of our galactic neighborhood.",
        "view_more": "https://www.google.com/search?q=Andromeda+constellation"
    },
    "Aquila": {
        "description": "Aquila is a constellation in the northern sky, symbolizing an eagle. Its bright stars and prominent position make it easy to locate during the summer months.",
        "mythology": "In Greek mythology, Aquila represents Zeus's eagle, which was tasked with carrying his thunderbolts. The eagle played a significant role in various myths, often symbolizing power and divine intervention. Aquila is also associated with the myth of Ganymede, whom Zeus transformed into an eagle to carry him to Olympus.",
        "facts": "The constellation’s brightest star, Altair, is part of the Summer Triangle, a prominent asterism formed by Altair, Deneb (in Cygnus), and Vega (in Lyra). Altair is one of the closest stars to Earth, located just 16.7 light-years away. Aquila also contains the star system Alshain, a binary system, and several deep-sky objects.",
        "best_visibility": "Summer (July to September), when it can be seen high in the evening sky across the northern hemisphere. This period offers the clearest views, especially when observing from rural areas with minimal light pollution.",
        "significance": "Aquila is significant for its association with the Summer Triangle, a highly visible asterism that marks the peak of the summer stargazing season. Its bright stars also make it important for celestial navigation and astronomical studies.",
        "view_more": "https://www.google.com/search?q=Aquila+constellation"
    },
    "Auriga": {
        "description": "Auriga, also known as the charioteer, is a bright constellation in the northern sky, often associated with mythological figures. Its shape is reminiscent of a charioteer holding the reins of a team of horses.",
        "mythology": "The charioteer is often linked to the myth of Erichthonius, an ancient king of Athens who is said to have invented the chariot. In some versions, Auriga represents him guiding the chariot of the gods. The constellation has also been connected to the myth of the Titan Phaethon, who lost control of the Sun's chariot.",
        "facts": "Auriga is home to Capella, the sixth-brightest star in the sky, which is a prominent yellow giant. The constellation also contains several star clusters, including M36, M37, and M38, which are popular targets for amateur astronomers. It is rich in double stars and other deep-sky objects, making it a fascinating area for exploration.",
        "best_visibility": "Winter (December to February), when Auriga is most visible in the northern hemisphere’s evening sky. The constellation is high in the sky during these months, making it ideal for stargazing in the winter season.",
        "significance": "Auriga’s position in the sky, along with its bright stars and star clusters, makes it an important feature for both amateur and professional astronomers. The constellation’s presence near the Milky Way also makes it a rich source of study in terms of star formation and galactic structure.",
        "view_more": "https://www.google.com/search?q=Auriga+constellation"
    },
    "Canis Major": {
        "description": "Canis Major, the Greater Dog, is a prominent southern constellation representing one of Orion's hunting dogs. It is easily recognizable due to its brightest star, Sirius, the brightest star in the night sky.",
        "mythology": "In Greek mythology, Canis Major is often associated with the hunting dogs of the hero Orion. The constellation is linked to several myths, including the story of Orion’s hunt and his connection with the gods. The dog is also said to be chasing the rabbit in the sky, Lepus.",
        "facts": "Sirius, the brightest star in the night sky, is located in Canis Major and is often called the Dog Star. It is part of a binary star system with a faint white dwarf companion. The constellation is home to several other stars and deep-sky objects, including the open star cluster Messier 41.",
        "best_visibility": "Winter (December to February), when Canis Major is visible in the evening sky in the southern hemisphere and lower latitudes of the northern hemisphere. Sirius is especially easy to spot during this period due to its brightness.",
        "significance": "Canis Major is significant for its association with Sirius, which was a key star in ancient navigation and agriculture. The star’s heliacal rising was used by ancient civilizations, including the Egyptians, to mark the start of the flooding of the Nile River, which was crucial for agriculture.",
        "view_more": "https://www.google.com/search?q=Canis+Major+constellation"
    },
    "Capricornus": {
        "description": "Capricornus, the sea-goat, is one of the twelve zodiac constellations. It lies between Aquarius and Sagittarius in the sky and is one of the faintest zodiac constellations. It is often depicted as a creature with the body of a goat and the tail of a fish, representing the dual nature of both land and water elements.",
        "mythology": "Capricornus is linked to the Greek god Pan, the god of the wild, shepherds, and flocks. According to myth, Pan transformed himself into a sea-goat to escape the monster Typhon, a creature of immense destructive power. This transformation allowed Pan to flee safely, and his new form became the symbol of Capricornus.",
        "facts": "This faint constellation is home to stars like Algedi and Dabih, which form part of the goat's body. Capricornus is also the source of the Capricornid meteor shower, which occurs annually in July. The constellation is also significant for marking the Winter Solstice, a time when the Sun enters Capricornus in the tropical zodiac.",
        "best_visibility": "Late Summer to Early Fall (August to October), when the constellation rises high in the sky in the evening, especially for observers in the Southern Hemisphere and parts of the Northern Hemisphere.",
        "significance": "Capricornus holds great astrological significance as a zodiac sign. It is associated with themes of discipline, responsibility, and practicality. Historically, it was also linked to agriculture, representing the transition between the harvest season and the winter solstice.",
        "view_more": "https://www.google.com/search?q=Capricornus+constellation"
    },
    "Cetus": {
        "description": "Cetus is a large constellation in the equatorial region, often depicted as a sea monster or whale. It is one of the largest constellations in the sky and is located near the celestial equator, making it visible from both hemispheres.",
        "mythology": "In Greek mythology, Cetus represents the sea monster sent by Poseidon to ravage the coast of Ethiopia. The monster was eventually slain by the hero Perseus, who rescued the princess Andromeda from its clutches. Cetus is a symbol of the eternal battle between good and evil, where the monster is vanquished by the hero.",
        "facts": "Cetus contains Mira, a famous variable star whose brightness changes dramatically over a 332-day cycle. This was one of the first stars discovered to be variable. The constellation also contains the barred spiral galaxy NGC 246 and several nebulae, adding to its astronomical significance.",
        "best_visibility": "Autumn (October to December), when Cetus is visible in the evening sky in the northern hemisphere and early in the night from the southern hemisphere.",
        "significance": "Cetus is significant for hosting Mira, one of the most famous variable stars. It provides valuable insights into stellar evolution, particularly the behavior of red giants in their later stages of life. The constellation also symbolizes perseverance and victory over adversity in mythology.",
        "view_more": "https://www.google.com/search?q=Cetus+constellation"
    },
    "Gemini": {
        "description": "Gemini, the twins, is one of the most prominent zodiac constellations. It represents the twin brothers Castor and Pollux, known for their adventures and inseparable bond. The constellation is located between Taurus and Cancer in the sky, and the bright stars Castor and Pollux mark the heads of the twins.",
        "mythology": "In Greek mythology, Castor and Pollux were the sons of Zeus and Leda. Castor was mortal, while Pollux was divine. The twins shared an unbreakable bond, and when Castor was killed in battle, Pollux asked Zeus to allow them to live together for eternity. Zeus granted their request, and they were transformed into stars, symbolizing loyalty, sacrifice, and brotherhood.",
        "facts": "Gemini is home to the bright stars Castor and Pollux, which are among the brightest in the night sky. The constellation is also the source of the Geminid meteor shower, one of the most active meteor showers, which takes place every December and is known for its spectacular display of meteors.",
        "best_visibility": "Winter (December to February), when Gemini is high in the sky during the evening, making it one of the most easily recognizable constellations in the night sky.",
        "significance": "Gemini is a key zodiac constellation, symbolizing duality, partnership, and brotherly devotion. Its presence in astrology is associated with versatility, communication, and adaptability. The constellation also plays a significant role in celestial navigation.",
        "view_more": "https://www.google.com/search?q=Gemini+constellation"
    },
    "Leo": {
        "description": "Leo, the lion, is a prominent zodiac constellation and one of the most easily recognizable constellations in the sky. It is located between Cancer and Virgo and represents the Nemean Lion defeated by Hercules in Greek mythology.",
        "mythology": "Leo is linked to the myth of the Nemean Lion, a creature with invulnerable skin that terrorized the land. The hero Hercules was tasked with killing the lion as one of his twelve labors. After defeating it, Hercules wore the lion’s skin as armor, and Zeus immortalized the lion by placing it among the stars.",
        "facts": "The brightest star in Leo is Regulus, a multiple star system that is also known as the 'Heart of the Lion.' The constellation is also home to the Leo Triplet, a group of three galaxies visible in the spring sky. The stars of Leo form a pattern that resembles a backward question mark, representing the lion’s mane and head.",
        "best_visibility": "Spring (March to May), when Leo is high in the sky during the evening and is easily visible across the northern hemisphere.",
        "significance": "As a zodiac constellation, Leo symbolizes strength, courage, and leadership. It is a prominent sign in astrology, representing those born under it as bold, charismatic, and confident. Leo’s association with the lion also signifies royalty and nobility.",
        "view_more": "https://www.google.com/search?q=Leo+constellation"
    },
    "Orion": {
        "description": "Orion is one of the most recognizable and famous constellations in the night sky. Located along the celestial equator, Orion is known for its distinctive shape, which resembles a hunter wielding a sword. It is situated between Taurus and Lepus and is visible from both hemispheres.",
        "mythology": "Named after Orion, the great hunter in Greek mythology, this constellation represents his exploits and heroic feats. According to myth, Orion was placed in the sky after his death by the gods, who honored his skills and bravery. Orion’s dog, Sirius, is represented by the constellation Canis Major.",
        "facts": "Orion contains several famous stars, including Betelgeuse (a red supergiant) and Rigel (a blue supergiant), both of which are among the brightest stars in the sky. The constellation also features the Orion Nebula, one of the brightest nebulae visible to the naked eye, which is a region of active star formation.",
        "best_visibility": "Winter (December to February), when Orion dominates the evening sky in the northern hemisphere and can be seen rising in the east after sunset.",
        "significance": "Orion is one of the most famous and culturally significant constellations. It is used in navigation, astronomy, and astrology and has deep cultural significance across many civilizations, symbolizing the hunter and the warrior. The constellation’s prominence in the sky makes it a favorite among stargazers and astronomers.",
        "view_more": "https://www.google.com/search?q=Orion+constellation"
    },
    "Sagittarius": {
        "description": "Sagittarius, the archer, is a zodiac constellation rich in deep-sky objects. It is symbolized by a centaur holding a bow and arrow, aiming towards the center of the Milky Way galaxy.",
        "mythology": "Sagittarius represents the centaur Chiron, a wise and noble figure in Greek mythology. Unlike other centaurs, Chiron was known for his kindness and knowledge, and he was often portrayed as a healer and teacher. He was placed among the stars by Zeus after his death, symbolizing wisdom and guidance.",
        "facts": "Sagittarius is home to the Galactic Center, the center of our Milky Way galaxy, and many other star clusters, nebulae, and deep-sky objects. Notable objects in the constellation include the Lagoon Nebula, the Trifid Nebula, and the Sagittarius A* supermassive black hole at the galaxy's center.",
        "best_visibility": "Summer (July to September), when the constellation is high in the sky during the night, especially in the southern hemisphere.",
        "significance": "Sagittarius has significance in both astronomy and astrology. As a zodiac sign, it symbolizes adventure, exploration, and philosophy. The constellation’s location near the Milky Way’s center makes it one of the most exciting regions in the night sky for astronomers.",
        "view_more": "https://www.google.com/search?q=Sagittarius+constellation"
    },
    "Taurus": {
        "description": "Taurus, the bull, is one of the prominent zodiac constellations. It is located in the northern sky, and its shape is often depicted as a bull with a V-shaped pattern representing its head. It is known for its bright star Aldebaran, which marks the eye of the bull, and the Pleiades and Hyades star clusters.",
        "mythology": "In Greek mythology, Taurus is associated with Zeus, who transformed into a magnificent white bull to abduct the princess Europa. Once Europa was on the back of the bull, Zeus swam across the sea to Crete, where he revealed his true identity. The bull was placed in the sky as a constellation by Zeus after the event.",
        "facts": "Taurus contains the Pleiades, an open star cluster that is visible to the naked eye, and the Hyades, the nearest open cluster to Earth. The bright star Aldebaran is part of the Hyades cluster, and it represents the bull’s eye. Taurus is also significant as a marker for the spring equinox in ancient times.",
        "best_visibility": "Winter (December to February), when Taurus is prominent in the evening sky, making it visible across much of the northern hemisphere.",
        "significance": "Taurus is significant for its bright stars, especially Aldebaran, and its star clusters, the Pleiades and Hyades. The constellation has long been associated with fertility and agriculture, as its appearance signaled the arrival of the planting season.",
        "view_more": "https://www.google.com/search?q=Taurus+constellation"
    },
    "Scorpius": {
        "description": "Scorpius is a zodiac constellation that represents a scorpion. It is easily recognizable due to its distinctive curved shape, which resembles a scorpion’s tail. Scorpius is located in the southern part of the sky and is one of the brightest constellations.",
        "mythology": "In Greek mythology, Scorpius represents the scorpion sent by the goddess Artemis to kill the hunter Orion. The two were placed in the sky as constellations, but they were positioned on opposite sides of the sky so that they would never be seen at the same time.",
        "facts": "Scorpius contains Antares, a bright red supergiant star located at the heart of the scorpion. The constellation is also home to several star clusters, including the Butterfly Cluster (M6) and the Ptolemy Cluster (M7), both of which are visible to the naked eye in dark skies.",
        "best_visibility": "Summer (July to September), when Scorpius is visible during the evening hours, especially in the southern hemisphere.",
        "significance": "Scorpius is a dramatic constellation, known for its bright stars and its role in ancient mythology. It is significant in astrology as a zodiac sign, associated with intense emotions, transformation, and passion.",
        "view_more": "https://www.google.com/search?q=Scorpius+constellation"
    },
    "Libra": {
        "description": "Libra, the scales, is a zodiac constellation that represents balance and justice. It is located in the southern sky and is one of the least conspicuous constellations. Its stars form an elegant shape that resembles a pair of scales.",
        "mythology": "In Greek mythology, Libra is associated with Themis, the goddess of justice. The constellation represents her scales, symbolizing balance, fairness, and the weighing of right and wrong. It is often connected with the judicial system and the pursuit of harmony.",
        "facts": "Libra contains the stars Zubenelgenubi and Zubeneschamali, which mark the two sides of the scales. These stars are part of the ecliptic, and Libra lies between Virgo and Scorpio in the zodiac. The constellation is often associated with the autumn equinox and the idea of equilibrium.",
        "best_visibility": "Spring (April to June), when Libra is visible in the evening sky in the southern hemisphere and during early summer in the northern hemisphere.",
        "significance": "Libra is symbolized by the scales, representing justice, fairness, and balance. It is one of the more abstract constellations, as it lacks a distinct mythological figure but is important for its connection to the concept of equality.",
        "view_more": "https://www.google.com/search?q=Libra+constellation"
    },
    "Pisces": {
        "description": "Pisces is a zodiac constellation that represents two fish tied together by a cord. The constellation is located in the northern sky and is one of the largest and faintest zodiac constellations. It symbolizes duality and the connection between the spiritual and the physical.",
        "mythology": "In Greek mythology, Pisces represents the fish into which the goddess Aphrodite and her son Eros transformed to escape the monster Typhon. The two fish were tied together by a cord to prevent them from drifting apart. They were later placed in the sky by Zeus as a symbol of their escape and protection.",
        "facts": "Pisces is notable for its connection to the Age of Pisces, which began around 1 AD and is associated with the Christian era. Although not bright, Pisces is home to the famous star system Alpha Piscium, and it marks the location of the vernal equinox in the tropical zodiac.",
        "best_visibility": "Autumn (October to December), when Pisces is visible in the night sky, especially in the northern hemisphere.",
        "significance": "Pisces is significant for its symbolism of water, duality, and the fish. In astrology, it represents emotional depth, intuition, and the transcendence of physical limitations. It is also the final sign of the zodiac.",
        "view_more": "https://www.google.com/search?q=Pisces+constellation"
    },
    "Virgo": {
        "description": "Virgo is the second-largest constellation and a zodiac constellation that represents a maiden or virgin. It is located in the southern part of the sky and is one of the most prominent and easily recognized constellations.",
        "mythology": "In Greek mythology, Virgo is often associated with Demeter, the goddess of the harvest, and her daughter Persephone. When Persephone was abducted by Hades, Demeter's grief caused the earth to become barren, and Virgo is said to represent the goddess’s sorrow and her connection to fertility and the harvest.",
        "facts": "Virgo contains Spica, a bright binary star system that is the 15th-brightest star in the sky. The constellation also features many galaxies, including the Sombrero Galaxy and the Virgo Cluster, which is a massive cluster of galaxies at the heart of the local universe.",
        "best_visibility": "Spring (April to June), when Virgo is visible in the evening sky, especially from the northern hemisphere.",
        "significance": "Virgo is significant for its connection to agriculture and the harvest. It is also associated with purity, fertility, and compassion in astrology, and it represents the maiden’s role as a nurturer and protector.",
        "view_more": "https://www.google.com/search?q=Virgo+constellation"
    },
    "Ursa Major": {
        "description": "Ursa Major, the Great Bear, is one of the largest and most recognizable constellations in the northern sky. It is famous for containing the Big Dipper, an asterism formed by seven bright stars that are often used to locate the North Star, Polaris.",
        "mythology": "In Greek mythology, Ursa Major represents Callisto, a beautiful nymph who was transformed into a bear by Hera. She was placed in the sky by Zeus as the constellation to honor her, though she was forever separated from her son Arcas, who became Ursa Minor.",
        "facts": "Ursa Major is home to the Big Dipper, which is an important navigational tool in the northern hemisphere. The constellation also contains other notable stars such as Dubhe and Merak, which help point to the North Star.",
        "best_visibility": "Spring (March to May), when Ursa Major is visible throughout the night, especially in the northern hemisphere.",
        "significance": "Ursa Major is significant for its role in navigation, as the Big Dipper points toward Polaris, the North Star. It is one of the most iconic constellations and has been a symbol of the Great Bear in various cultures.",
        "view_more": "https://www.google.com/search?q=Ursa+Major+constellation"
    },
    "Ursa Minor": {
        "description": "Ursa Minor, the Little Bear, is home to Polaris, the North Star, and is one of the most important constellations for navigation. It is located in the northern sky and contains the Little Dipper, an asterism formed by seven stars.",
        "mythology": "In Greek mythology, Ursa Minor represents Arcas, the son of Callisto, who was placed in the sky as a bear by Zeus. Arcas was transformed into a star to be reunited with his mother, Callisto, who had been turned into the larger bear, Ursa Major.",
        "facts": "The most notable feature of Ursa Minor is Polaris, which has been used for navigation for centuries. Polaris is part of the Little Dipper, and its position near the celestial pole makes it a fixed point in the sky, marking true north.",
        "best_visibility": "Year-round in the Northern Hemisphere, as Ursa Minor is circumpolar and visible throughout the year from most locations in the northern sky.",
        "significance": "Ursa Minor is significant for its role in navigation due to the constant position of Polaris, which has guided explorers and travelers for millennia. The constellation also symbolizes the eternal bond between Arcas and his mother Callisto.",
        "view_more": "https://www.google.com/search?q=Ursa+Minor+constellation"
    },
    "Lyra": {
        "description": "Lyra is a small, compact constellation that represents a lyre or harp. Located in the northern sky, it is one of the most notable constellations for its bright star Vega. Lyra's shape resembles a small parallelogram, which represents the frame of the lyre.",
        "mythology": "Lyra is associated with Orpheus, the legendary Greek musician and poet. According to myth, Orpheus played such beautiful music on his lyre that even the gods were moved. After his death, Zeus placed the lyre in the sky as a constellation in memory of Orpheus.",
        "facts": "Lyra contains Vega, the fifth-brightest star in the sky, and it is part of the Summer Triangle, which includes the stars Deneb and Altair. The constellation also contains the Ring Nebula (M57), a famous planetary nebula that is visible through telescopes.",
        "best_visibility": "Summer (July to September), when Lyra is visible high in the northern sky during the late evening hours.",
        "significance": "Lyra is known for its bright star Vega, which has been used as a standard for astronomical brightness, and for the Ring Nebula, a fascinating object of study for astronomers.",
        "view_more": "https://www.google.com/search?q=Lyra+constellation"
    },
    "Cassiopeia": {
        "description": "Cassiopeia is a prominent W-shaped constellation in the northern sky. It is easily recognizable due to its distinct 'W' or 'M' shape, which represents the mythical queen Cassiopeia, seated on her throne. It is one of the most notable constellations in the northern hemisphere.",
        "mythology": "Named after Queen Cassiopeia, who, in Greek mythology, boasted about her beauty, claiming she was more beautiful than the Nereids, sea nymphs. Offended by this arrogance, Poseidon sent a sea monster to threaten her kingdom. As punishment, Cassiopeia was placed in the sky, bound to circle the North Pole forever.",
        "facts": "Cassiopeia contains the Cassiopeia A supernova remnant, a powerful source of radio waves. The constellation is also home to several bright stars and is often used in navigation due to its distinct shape, particularly as it is located near the North Star, Polaris.",
        "best_visibility": "Autumn (September to November), when Cassiopeia is high in the northern sky during the evening hours.",
        "significance": "Cassiopeia is significant for its distinctive shape, which makes it one of the easiest constellations to recognize. It has been important for navigation and is often used as a reference point in the sky.",
        "view_more": "https://www.google.com/search?q=Cassiopeia+constellation"
    },
    "Perseus": {
        "description": "Perseus is a northern constellation named after the Greek hero Perseus, known for his bravery and legendary feats. It lies near the Milky Way and contains some of the brightest stars in the sky, including Algol, a variable star.",
        "mythology": "In Greek mythology, Perseus is the hero who slew Medusa, one of the Gorgon sisters, whose gaze could turn people to stone. He is also credited with rescuing Andromeda from a sea monster, Cetus, and later marrying her. Perseus is celebrated for his heroic deeds and strength.",
        "facts": "Perseus contains the famous variable star Algol, also known as the Demon Star, due to its periodic dimming and brightening. It is also the source of the Perseid meteor shower, one of the most well-known meteor showers of the year.",
        "best_visibility": "Autumn (September to November), when Perseus is visible in the evening sky, particularly in the northern hemisphere.",
        "significance": "Perseus is significant for its association with the legendary hero and its historical importance as the origin of the Perseid meteor shower, a yearly event that delights stargazers around the world.",
        "view_more": "https://www.google.com/search?q=Perseus+constellation"
    },
    "Draco": {
        "description": "Draco, the dragon, is a large, northern circumpolar constellation, meaning it is visible all year round in the Northern Hemisphere. Its shape is quite serpentine, resembling a dragon curling around the North Pole.",
        "mythology": "In Greek mythology, Draco represents the dragon that was slain by Hercules as part of his Twelve Labors. The dragon guarded the golden apples of the Hesperides, and its placement in the sky was a reward for its protection of the treasure.",
        "facts": "Draco contains Thuban, which was once the North Star, and the Cat's Eye Nebula (NGC 6543), one of the most famous planetary nebulae. The constellation is very large, stretching across a significant portion of the northern sky.",
        "best_visibility": "Year-round in the Northern Hemisphere, as Draco is a circumpolar constellation. It is especially visible during the summer months.",
        "significance": "Draco is significant for its historical role in navigation, as it once hosted the North Star, Thuban, before the Earth's axis shifted. It is also known for the Cat's Eye Nebula, a popular object for amateur astronomers to observe.",
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
                           view_more=constellation_info.get("view_more", "#"))


# Run the Flask app
# Run the Flask app
if __name__ == '__main__':
    # Ensure the uploads directory exists
    os.makedirs('static/uploads', exist_ok=True)
    
    # Get the port from the environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app on the correct port and make it accessible externally
    app.run(debug=True, host='0.0.0.0', port=port)
