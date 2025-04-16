import os
import json
import face_recognition
import cv2
import spacy
from imutils import build_montages
import numpy as np
from duckduckgo_search import DDGS
import requests
from io import BytesIO
from PIL import Image

# Load NLP model
nlp = spacy.load("es_core_news_sm")

# Path to face image folder and database
FACE_FOLDER = r"D:\City College Dropbox\Melanie Victoria Adler\Personal\Face_rekognition_Infobae\faces"
DB_PATH = "faces_db.json"

# Global variable to hold the face image data
face_images = []
image_paths = []

def show_face_montage(matches, person_name):
    global face_images, image_paths
    face_images = []
    image_paths = []

    print(f"🔍 Montage for {person_name} – matches: {len(matches)}")

    for i, match in enumerate(matches):
        try:
            path = match["path"]
            image = cv2.imread(path)
            if image is None:
                print(f"⚠️ Could not load image at {path}")
                continue
            image = cv2.resize(image, (96, 96))
            face_images.append(image)
            image_paths.append(path)  # Store the path to each image for later use
        except Exception as e:
            print(f"❌ Error processing match {i}: {e}")

    if not face_images:
        print(f"No images to display for {person_name}")
        return

    montage = build_montages(face_images, (96, 96), (5, 5))[0]
    title = f"Possible Matches: {person_name}"

    # Display the montage and set up mouse callback
    cv2.imshow(title, montage)
    cv2.setMouseCallback(title, mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def search_images(name, max_results=3):
    with DDGS() as ddgs:
        results = ddgs.images(keywords=name)
        image_urls = []
        for result in results:
            if 'image' in result:
                image_urls.append(result['image'])
                if len(image_urls) >= max_results:
                    break
        return image_urls

def mouse_callback(event, x, y, flags, param):
    global face_images, image_paths
    if event == cv2.EVENT_LBUTTONDOWN:  # Check if left mouse button is clicked
        # Calculate the index of the clicked image based on the x and y coordinates
        image_width = 96
        image_height = 96
        montage_width = 5 * image_width  # 5 images per row
        montage_height = 5 * image_height  # 5 images per column

        row = y // image_height
        col = x // image_width

        # Ensure the index doesn't exceed the number of images in the montage
        index = row * 5 + col
        if index < len(image_paths):
            print(f"🔍 You clicked on image: {image_paths[index]}")
            # Load and display the clicked image
            show_face_image(image_paths[index])

def show_face_image(path):
    image = cv2.imread(path)
    if image is None:
        print(f"⚠️ Could not load image at {path}")
        return
    cv2.imshow("Selected Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_or_create_face_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'r') as f:
            face_db = json.load(f)
    else:
        face_db = {}

    # Ensure entries are lists (backward compatibility)
    for k, v in face_db.items():
        if isinstance(v, dict):
            face_db[k] = [v]

    # Get already processed file paths
    known_paths = {entry['path'] for entries in face_db.values() for entry in entries}

    # Process only new images
    for filename in os.listdir(FACE_FOLDER):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(FACE_FOLDER, filename)

        if path in known_paths:
            continue  # Already processed

        print(f"🆕 Processing new face: {filename}")
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            name = os.path.splitext(filename)[0].replace("_", " ").title()
            entry = {
                "encoding": encodings[0].tolist(),
                "path": path
            }
            face_db.setdefault(name, []).append(entry)
        else:
            print(f"⚠️ No face found in {filename}, skipping.")

    # Save updated database
    with open(DB_PATH, 'w') as f:
        json.dump(face_db, f)

    return face_db

def extract_people_from_text(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "PER"]  # when using the model in english look for PERSON

def find_all_matching_faces(name, db):
    target_name = name.lower()
    matches = []
    for known_name, entries in db.items():
        if target_name in known_name.lower():
            matches.extend(entries)
    return matches


def search_and_add_face(person, db):
    print(f"🔍 Searching for {person} online...")

    # Search for images of the person
    image_urls = search_images(person)

    if not image_urls:
        print(f"❌ No images found for {person}")
        return False

    # For each image URL, try to download and add it to the database
    for url in image_urls:
        try:
            print(f"Downloading image from {url}")
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img.save(f"downloaded_{person}.jpg")  # Save the image temporarily

            # Add the downloaded image to the database
            new_entry = {
                "encoding": face_recognition.face_encodings(face_recognition.load_image_file(f"downloaded_{person}.jpg"))[0].tolist(),
                "path": f"downloaded_{person}.jpg"
            }

            db.setdefault(person, []).append(new_entry)
            print(f"✅ Added image for {person} to database")
            return True

        except Exception as e:
            print(f"❌ Error adding face for {person}: {e}")
            continue

    return False

def merge_similar_names(names):
    merged = []
    for name in names:
        if any(name in full_name or full_name in name for full_name in merged):
            continue
        merged.append(name)
    return merged

def main(text, status_callback=None):
    def update(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    db = load_or_create_face_db()
    people = extract_people_from_text(text)
    update(f"🔍 Detected people (raw): {people}")

    # Normalize and deduplicate based on substrings
    people = sorted(set(people), key=lambda x: -len(x))  # Sort longest first
    unique_people = merge_similar_names(people)

    update(f"✅ Unique people (smart filtered): {unique_people}")

    for person in unique_people:
        matches = find_all_matching_faces(person, db)
        if matches:
            update(f"📸 Found {len(matches)} match(es) for {person}!")
            show_face_montage(matches, person)
        else:
            update(f"❌ No match found for {person} in local DB.")
            update(f"🌐 Searching online for {person}...")
            if search_and_add_face(person, db):
                matches = find_all_matching_faces(person, db)
                show_face_montage(matches, person)
                update(f"📸 Done showing montage for {person}.")
            else:
                update(f"😞 Could not find any usable images for {person} online.")

    update("✅ Done processing all names.")
