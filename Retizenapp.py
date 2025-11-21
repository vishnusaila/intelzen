import json
import time
import re
import datetime
import traceback
import boto3
import io
import threading
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS 
import schedule
import requests 
import os
import uuid 

# ---------------- CONFIGURATION ----------------
# The following variables attempt to load from the environment.
# Since this script runs locally, ensure your environment variables are set, 
# or the fallbacks are used.


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
GEOCODE_API_KEY = os.getenv("GEOCODE_API_KEY")
 
# ----------------------------------------------

# ---------------- FLASK APP SETUP ----------------
app = Flask(__name__)
CORS(app)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="gemini-2.5-pro")

# ---------------- PROMPT ----------------
prompt = """
You are an AI model that classifies uploaded media (image, audio, or video).
Classify each incident into one of these categories:
Law Enforcement, Public Safety, Behavioral, Junk, or Unknown.

Guidelines for Classification and assigning priority level:
1. Law Enforcement: [Details omitted for brevity]
2. Public Safety: [Details omitted for brevity]
3. Behavioral: [Details omitted for brevity]
4. Junk: [Details omitted for brevity]
5. Unknown: [Details omitted for brevity]

Return output as JSON only:
{
    "reason": "<Short reason>",
    "label": "<Law Enforcement | Public Safety | Behavioral | Junk | Unknown>",
    "Priority": "1 | 2 | 3 | 4"
}
Now classify the following content:
"""

# ---------------- DICTIONARIES ----------------
mimetype_dict = {
    "png": "image/png", "mp4": "video/mp4", "mov": "video/quicktime", 
    "heic": "image/heic", "jpg": "image/jpeg", "jpeg": "image/jpeg", 
    "mp3": "audio/mpeg",
}

department_dict = {
    "Public Safety": "Public Safety",
    "Law Enforcement": "Legal",
    "Behavioral": "Crime",
    "Junk": "None",
    "Unknown": "Unknown",
}

# ---------------- S3 HELPERS ----------------
def get_s3_resource():
    return boto3.resource(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

def save_result_to_s3(userid, caseid, result_data):
    """Save classification result to S3 bucket."""
    try:
        s3 = get_s3_resource()
        key = f"{userid}/{caseid}/Result/result.json"
        s3.Bucket(S3_BUCKET_NAME).put_object(
            Key=key, Body=json.dumps(result_data).encode("utf-8")
        )
        print(f"✅ Uploaded result.json to s3://{S3_BUCKET_NAME}/{key}")
    except Exception as e:
        print(f"❌ Failed to upload result.json: {e}")

# ---------------- GEOCoding FUNCTION ----------------
def get_location_from_coords(latitude, longitude):
    """Reverse geocodes the coordinates using the Google Maps Geocoding API."""
    if latitude == 0.0 and longitude == 0.0:
        return "N/A (No Coords)"
    
    # Use a dummy key if the actual key is not set to prevent crashing
    if not GEOCODE_API_KEY or GEOCODE_API_KEY == "YOUR_GOOGLE_MAPS_GEOCODE_API_KEY":
        return f"Coords: {latitude}, {longitude} (Geocoding API Key Missing)"

    print(f"🗺️ Attempting to geocode coords: {latitude}, {longitude}...")
    
    try:
        url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={latitude},{longitude}&key={GEOCODE_API_KEY}"
        response = requests.get(url, timeout=10).json()
        
        if response.get("status") == "OK" and response.get("results"):
            # Attempt to find locality/city in the results
            for result in response["results"]:
                for component in result["address_components"]:
                    if "locality" in component["types"]:
                        return component["long_name"]
                    if "postal_town" in component["types"] or "administrative_area_level_2" in component["types"]:
                        return component["long_name"]

            return response["results"][0]["formatted_address"]
        else:
            print(f"❌ Geocoding API failed. Status: {response.get('status')}")
            return f"Coords: {latitude}, {longitude} (API Error: {response.get('status')})"

    except requests.exceptions.RequestException as e:
        print(f"❌ Geocoding Request failed: {e}")
        return f"Coords: {latitude}, {longitude} (Request Error)"

# ---------------- GEMINI CLASSIFIER ----------------
def safe_generate(model, content):
    retries = 3
    for attempt in range(retries):
        try:
            return model.generate_content(content)
        except Exception as e:
            if attempt < retries - 1:
                delay = 5 * (attempt + 1)
                time.sleep(delay)
            else:
                raise

def score(file_stream, mimetype):
    """Uploads media to Gemini and returns classification."""
    file_stream.seek(0)
    # Using positional argument for file stream
    sample_file = genai.upload_file(file_stream, mime_type=mimetype) 

    while sample_file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(10)
        sample_file = genai.get_file(sample_file.name)

    if sample_file.state.name == "FAILED":
        raise ValueError("File processing failed on Gemini side")

    response = safe_generate(model, [prompt, sample_file])
    sample_file.delete()

    match = re.findall(r"\{[\s\S]*\}", response.text)
    if not match:
        raise ValueError("Gemini response missing JSON output")

    try:
        return json.loads(match[0])
    except Exception as e:
        raise ValueError(f"Invalid JSON format in Gemini response: {e}")

# ---------------- CLASSIFIER CORE ----------------
def classifier(userid, caseid):
    """Main classification logic using Gemini + S3 integration."""
    print(f"\n🧠 Starting classification for {userid}/{caseid}...")
    s3_dir_path = f"{userid}/{caseid}/Pending"
    s3_case_path = f"{userid}/{caseid}"

    final_response = {
        "Department": "None",
        "label": "Pending",
        "reason": "Classification started.",
        "Priority": "4",
    }
    s3 = get_s3_resource()
    bucket = s3.Bucket(S3_BUCKET_NAME)

    try:
        # Find media file
        file_paths = [
            obj.key for obj in bucket.objects.filter(Prefix=s3_dir_path)
            if not obj.key.endswith("/")
        ]

        if not file_paths:
            raise FileNotFoundError(f"No media found under {s3_dir_path}")

        file_path = file_paths[0]
        ext = file_path.split(".")[-1].lower()
        mimetype = mimetype_dict.get(ext)
        
        if not mimetype:
            raise ValueError(f"Unsupported file extension: {ext}")

        # Download file to stream
        obj = bucket.Object(file_path)
        file_stream = io.BytesIO()
        obj.download_fileobj(file_stream)

        # Perform classification
        response_dict = score(file_stream, mimetype)
        label = response_dict.get("label", "Unknown")
        response_dict["Department"] = department_dict.get(label, "Unknown")
        final_response = response_dict

        # ---- Update Master Data ----
        try:
            mstr_obj = s3.Object(S3_BUCKET_NAME, "Master/mockdata_with_category_images.json")
            mstr_data = json.load(mstr_obj.get()["Body"])

            meta_obj = s3.Object(S3_BUCKET_NAME, f"{s3_case_path}/metadata.json")
            meta_data = json.load(meta_obj.get()["Body"])

            # --- LOCATION RESOLUTION LOGIC ---
            reported_city = meta_data.get("city")
            latitude = float(meta_data.get("latitude", 0.0))
            longitude = float(meta_data.get("longitude", 0.0))

            is_city_valid = reported_city and reported_city.strip().lower() not in ["n/a", "undefined", "unknown"]
            
            if is_city_valid:
                resolved_location = reported_city
            else:
                resolved_location = get_location_from_coords(latitude, longitude)
            
            # Safely parse date
            date_str = meta_data.get("date", datetime.datetime.now().isoformat())
            try:
                date_object = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ").date()
            except ValueError:
                date_object = datetime.date.today()
                
            closed_date_str = (date_object + datetime.timedelta(days=10)).strftime("%Y-%m-%d")
            media_url = f"https://s3-{AWS_REGION}.amazonaws.com/{S3_BUCKET_NAME}/{file_path}"
            media_type = obj.content_type.split("/")[0] if obj.content_type else "unknown"

            new_record = {
                "Date": date_object.strftime("%Y-%m-%d"),
                "User ID": userid,
                "Case ID": caseid,
                "Location": resolved_location,
                "latitude": latitude,
                "longitude": longitude,
                "Category": final_response["label"],
                "Status": "Open",
                "Priority": final_response["Priority"],
                "ClosedDate": closed_date_str,
                "MediaFiles": [{"url": media_url, "type": media_type}],
            }

            # Update existing record or append new record
            found = False
            for i, rec in enumerate(mstr_data):
                if rec.get("Case ID") == caseid and rec.get("User ID") == userid:
                    mstr_data[i] = new_record # Overwrite the 'Pending' entry
                    found = True
                    break
            
            if not found:
                mstr_data.append(new_record)
            
            mstr_obj.put(Body=json.dumps(mstr_data, indent=4).encode("utf-8"))
            print(f"✅ Masterdata updated for case {caseid}. Found/Appended: {found}")

        except Exception as mstr_err:
            print(f"❌ Masterdata update failed: {mstr_err}")

    except Exception as err:
        print(f"❌ Error in classifier: {err}")
        final_response = {
            "Department": "Error",
            "label": "Error Raised in Model API",
            "reason": str(err),
            "Priority": "4",
        }

    save_result_to_s3(userid, caseid, final_response)
    return json.dumps(final_response)

# ---------------- SCHEDULING LOGIC ----------------
# (Scheduling logic is outside the scope of the Flask request flow but kept for completeness)
def find_and_classify_pending_cases():
    """Scans S3 for new 'Pending' cases that don't have a 'Result' yet."""
    # (Implementation omitted for brevity)
    pass

def run_scheduler():
    # (Implementation omitted for brevity)
    pass

# ---------------- FLASK ROUTES ----------------

@app.route("/api/upload/<userid>/<category>", methods=["POST"])
def handle_file_upload(userid, category):
    """
    Handles the initial file upload and metadata saving to S3.
    It returns the case ID immediately.
    """
    if 'files' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    uploaded_files = request.files.getlist('files')
    if not uploaded_files:
        return jsonify({"error": "No files selected for upload"}), 400

    caseid = str(uuid.uuid4()).split('-')[0]
    s3 = get_s3_resource()
    
    try:
        # Save files to S3's pending directory
        for file in uploaded_files:
            filename = file.filename
            s3_key = f'{userid}/{caseid}/Pending/{filename}'
            s3.Bucket(S3_BUCKET_NAME).upload_fileobj(file, s3_key)
        
        # Save initial metadata (including the 'Pending' category)
        metadata = {
            'description': request.form.get('description'),
            'status': request.form.get('status'),
            'date': datetime.datetime.now().isoformat(), # Use current time for accurate record keeping
            'state': request.form.get('state'),
            'latitude': request.form.get('latitude', '0.0'),
            'longitude': request.form.get('longitude', '0.0'),
            'city': request.form.get('city'),
            'initial_category': category # Should be 'Pending' from the frontend
        }
        s3.Object(S3_BUCKET_NAME, f'{userid}/{caseid}/metadata.json').put(
            Body=json.dumps(metadata)
        )

        # Append initial record to Masterdata immediately (Category: Pending)
        try:
            mstr_obj = s3.Object(S3_BUCKET_NAME, "Master/mockdata_with_category_images.json")
            mstr_data = json.load(mstr_obj.get()["Body"])
            
            # Create a simplified pending record
            pending_record = {
                "Date": datetime.date.today().strftime("%Y-%m-%d"),
                "User ID": userid,
                "Case ID": caseid,
                "Location": metadata.get("city", "Pending Location"),
                "latitude": float(metadata.get("latitude", 0.0)),
                "longitude": float(metadata.get("longitude", 0.0)),
                "Category": "Pending", # Hardcode Pending so the classifier can overwrite it
                "Status": "Open",
                "Priority": "4",
                "ClosedDate": (datetime.date.today() + datetime.timedelta(days=10)).strftime("%Y-%m-%d"),
                "MediaFiles": [{"url": f"https://s3-{AWS_REGION}.amazonaws.com/{S3_BUCKET_NAME}/{s3_key}", "type": mimetype_dict.get(ext, "image")}]
            }
            mstr_data.append(pending_record)
            mstr_obj.put(Body=json.dumps(mstr_data, indent=4).encode("utf-8"))
            print(f"✅ Initial Pending record appended to Masterdata for case {caseid}.")

        except Exception as e:
            print(f"⚠️ Failed to write initial pending record to Masterdata: {e}")
            
        return jsonify({"message": "Upload successful, classification ready", "caseId": caseid}), 200

    except Exception as e:
        print(f"Error during file upload: {e}")
        return jsonify({"error": f"Internal server error during S3 upload: {str(e)}", "details": traceback.format_exc()}), 500

@app.route("/api/upload_complete", methods=["POST"])
def trigger_classification():
    """Endpoint called by the frontend to run the AI classifier."""
    data = request.get_json()
    userid = data.get("userid")
    caseid = data.get("caseid")

    if not userid or not caseid:
        return jsonify({"error": "Missing userid or caseid in JSON body"}), 400

    # Execute classification in a background thread
    threading.Thread(target=classifier, args=(userid, caseid), daemon=True).start()

    return jsonify({"message": "Classification processing started successfully in background."}), 200

@app.route("/api/Mastercases/Master", methods=["GET"])
def serve_master_data():
    """Serve master cases."""
    try:
        s3 = get_s3_resource()
        mstr_obj = s3.Object(S3_BUCKET_NAME, "Master/mockdata_with_category_images.json")
        mstr_data_body = mstr_obj.get()["Body"].read().decode('utf-8')
        data_list = json.loads(mstr_data_body)
        return jsonify({"data": data_list, "message": "Master cases fetched successfully"}), 200
    except Exception as e:
        return jsonify({"error": "Failed to retrieve master case data", "details": str(e)}), 500

@app.route('/service-worker.js')
def service_worker():
    return "", 204

@app.route("/api/cases-with-images-json/<userid>", methods=["GET"])
def get_cases_for_user(userid):
    # This route is maintained for compatibility if the frontend still uses it
    return serve_master_data()

# ---------------- MAIN DRIVER ----------------
if __name__ == "__main__":
    # Start the Flask server
    print("🚀 Retizen Flask backend running at http://127.0.0.1:3001/")
    app.run(port=3001, debug=True, use_reloader=False)
