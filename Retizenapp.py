import json
import time
import re
import datetime
import boto3
import io
import threading
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import schedule
import requests
from botocore.exceptions import ClientError
import os
import traceback 

# --- 1. CONFIGURATION ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
GEOCODE_API_KEY = os.getenv("GEOCODE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not AWS_ACCESS_KEY_ID or not GEMINI_API_KEY:
    raise ValueError("FATAL: AWS or GEMINI credentials missing. Check configuration.")
# ------------------------

# --- 3. PROMPT AND DICTIONARIES ---
prompt = """
You are an AI model that classifies uploaded media (image, audio, or video).
Classify each incident into one of these categories:
Law Enforcement, Public Safety, Behavioral, Junk, or Unknown.

Return output as JSON only:
{
    "reason": "<Short reason>",
    "label": "<Law Enforcement | Public Safety | Behavioral | Junk | Unknown>",
    "Priority": "1 | 2 | 3 | 4"
}
"""

mimetype_dict = {
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "mp4": "video/mp4", "mov": "video/quicktime", "heic": "image/heic",
    "mp3": "audio/mpeg",
}

department_dict = {
    "Public Safety": "Public Safety", "Law Enforcement": "Legal",
    "Behavioral": "Crime", "Junk": "None", "Unknown": "Unknown",
}

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(
    model_name="gemini-2.5-pro",
    generation_config=genai.types.GenerationConfig(
        response_mime_type="application/json",
    )
)

# --- 4. S3 CLIENTS AND HELPERS ---
def get_s3_resource():
    """Returns a boto3 S3 Resource object."""
    return boto3.resource(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

def get_s3_client():
    """Returns a boto3 S3 Client object."""
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

def get_location_from_coords(latitude, longitude):
    """Reverse geocodes the coordinates using the Google Maps Geocoding API."""
    if latitude == 0.0 and longitude == 0.0 or not GEOCODE_API_KEY:
        return "N/A (No Coords)"
    
    try:
        url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={latitude},{longitude}&key={GEOCODE_API_KEY}"
        response = requests.get(url).json()
        
        if response.get("status") == "OK" and response.get("results"):
            for result in response["results"]:
                for component in result["address_components"]:
                    if "locality" in component["types"]:
                        return component["long_name"]
            return response["results"][0]["formatted_address"]
        return f"Coords: {latitude}, {longitude} (API Error: {response.get('status')})"
    except requests.exceptions.RequestException:
        return f"Coords: {latitude}, {longitude} (Request Error)"

def safe_generate(model, prompt_parts):
    """Safety wrapper with retries for Gemini API calls."""
    retries = 3
    for attempt in range(retries):
        try:
            return model.generate_content(prompt_parts)
        except Exception as e:
            print(f"⚠️ Gemini error (Attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                raise

def score(file_stream, mimetype, custom_prompt=None):
    """Uploads media to Gemini and returns classification."""
    file_stream.seek(0)
    sample_file = genai.upload_file(file_stream, mime_type=mimetype)
    
    final_prompt = custom_prompt if custom_prompt else prompt

    print(f"🧠 Uploading file to Gemini...", end='', flush=True)
    while sample_file.state.name == "PROCESSING":
        print('.', end='', flush=True)
        time.sleep(10)
        sample_file = genai.get_file(sample_file.name)
    print(" Complete.")

    if sample_file.state.name == "FAILED":
        raise ValueError(sample_file.state.name)

    response = safe_generate(model, [final_prompt, sample_file]) 
    sample_file.delete()

    try:
        return json.loads(response.text.strip())
    except Exception as e:
        raise ValueError(f"Invalid JSON format in Gemini response: {e}. Raw: {response.text}")

# --- NEW: USER-SPECIFIC SEQUENTIAL CASE ID LOGIC ---
def get_next_caseid(userid):
    """Fetches the next sequential case ID for a specific user and increments the counter."""
    s3_resource = get_s3_resource()
    counter_key = "Master/user_case_counter.json" 
    
    try:
        counter_obj = s3_resource.Object(S3_BUCKET_NAME, counter_key)
        counter_data = json.load(counter_obj.get()["Body"])
    except Exception as e:
        print(f"WARN: User counter file missing or error ({e}). Initializing empty counter.")
        counter_data = {}
        
    current_id = counter_data.get(userid, 1)
    
    next_id = current_id + 1
    
    counter_data[userid] = next_id
    
    s3_resource.Object(S3_BUCKET_NAME, counter_key).put(
        Body=json.dumps(counter_data).encode("utf-8"),
        ContentType='application/json'
    )
    
    return str(current_id)

# --- 5. CLASSIFIER CORE FUNCTION ---
def classifier(userid, caseid):
    """Main classification logic. This function UPDATES the existing Master record."""
    print(f"\n🧠 Starting classification for {userid}/{caseid}...")
    s3_dir_path = f"{userid}/{caseid}/Pending"
    s3_case_path = f"{userid}/{caseid}"
    result_key = f"{s3_case_path}/result.json"

    final_response = {
        "Department": "None", "label": "Junk",
        "reason": "Setup/Initial Error", "Priority": "4",
    }
    
    s3_resource = get_s3_resource()
    bucket = s3_resource.Bucket(S3_BUCKET_NAME)
    mstr_key = "Master/mockdata_with_category_images.json"
    mstr_obj = s3_resource.Object(S3_BUCKET_NAME, mstr_key)
    record_index = -1
    mstr_data = [] 
    resolved_location = "Unknown Location"
    
    try:
        # Load the Master data list
        mstr_data = json.load(mstr_obj.get()["Body"])
        
        # 1. Find the index of the pending record and extract location
        for i, record in enumerate(mstr_data):
            if str(record.get("Case ID")) == caseid and str(record.get("User ID")) == userid:
                record_index = i
                resolved_location = record.get("Location", "Unknown Location") 
                break
        
        if record_index == -1:
            raise FileNotFoundError("Pending record not found in Master data list. Cannot update.")
            
        # 2. Check if result.json already exists (redundant check, but safer)
        try:
            s3_resource.Object(S3_BUCKET_NAME, result_key).load()
            print(f"⏭️ Case {userid}/{caseid} already classified. Skipping.")
            return json.dumps({"message": "Already classified"})
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code not in ('NoSuchKey', '404'):
                raise 
        except:
            pass
            
        # 3. Get Media File for Classification
        file_paths = [
            obj.key for obj in bucket.objects.filter(Prefix=s3_dir_path)
            if not obj.key.endswith("/") and "." in obj.key.split("/")[-1]
        ]
        if not file_paths:
            raise FileNotFoundError(f"No media found under {s3_dir_path}")

        file_path = file_paths[0]
        ext = file_path.split(".")[-1].lower()
        
        # 4. Download File & Classify
        object_data = bucket.Object(file_path)
        file_stream = io.BytesIO()
        object_data.download_fileobj(file_stream)

        mimetype = mimetype_dict.get(ext.lower(), 'application/octet-stream')
        
        # --- MODIFIED PROMPT WITH LOCATION CONTEXT ---
        location_context_prompt = (
            f"{prompt}\n\n[CONTEXT: The incident occurred at or near: {resolved_location}]"
        )
        print(f"🗺️ Classifying with location: {resolved_location}")
        
        response_dict = score(file_stream, mimetype, custom_prompt=location_context_prompt)
        
        label = response_dict.get("label", "Unknown")
        final_response = response_dict
        
        print("The label is......", label)
        
        # 5. Update the Master Record with Classification Results
        if label.lower() not in ("junk", "unknown", "error"):
            final_response["Department"] = department_dict.get(label, "Unknown")
            
        mstr_data[record_index]["Category"] = final_response["label"]
        mstr_data[record_index]["Status"] = "Classified"
        mstr_data[record_index]["Priority"] = final_response.get("Priority", "4")
        mstr_data[record_index]["Department"] = final_response.get("Department", "None") 
        
        mstr_obj.put(Body=json.dumps(mstr_data, indent=4).encode("utf-8"))
        print(f"✅ Successfully UPDATED Masterdata for Case {caseid}.")

    except Exception as err:
        print(f"❌ Error in classifier: {err}")
        final_response = {"label": "Classification Error", "reason": str(err), "Priority": "4"}
        
        if record_index != -1:
            mstr_data[record_index]["Category"] = "Classification Error"
            mstr_data[record_index]["Status"] = "Error"
            mstr_data[record_index]["Priority"] = final_response["Priority"]
            mstr_obj.put(Body=json.dumps(mstr_data, indent=4).encode("utf-8"))

    # 6. Write Final result.json
    result_obj = bucket.Object(f"{s3_case_path}/result.json")
    result_obj.put(Body=json.dumps(final_response).encode("utf-8"))
    
    return json.dumps(final_response)

# --- 6. SCHEDULER LOGIC ---

def find_and_classify_pending_cases():
    """Scans S3 for new 'Pending' cases that don't have a 'result.json' yet."""
    s3_resource = get_s3_resource()
    bucket = s3_resource.Bucket(S3_BUCKET_NAME)

    try:
        all_objects = bucket.objects.filter(Prefix="")
        pending_cases = set()
        cases_with_pending_media = set()
        
        for obj in all_objects:
            parts = obj.key.split('/')
            if len(parts) >= 3:
                userid = parts[0]
                caseid = parts[1]
                
                if parts[2] == 'Pending' and not obj.key.endswith('/'):
                    cases_with_pending_media.add((userid, caseid))
        
        for userid, caseid in cases_with_pending_media:
            result_key = f"{userid}/{caseid}/result.json"
            try:
                get_s3_resource().Object(S3_BUCKET_NAME, result_key).load()
            except:
                if (userid, caseid) not in pending_cases:
                    pending_cases.add((userid, caseid))

        if pending_cases:
            print(f"Found {len(pending_cases)} case(s) to classify: {pending_cases}")
            for userid, caseid in pending_cases:
                threading.Thread(target=classifier, args=(userid, caseid), daemon=True).start()

    except Exception as e:
        print(f"❌ Error in scheduled job: {e}")


def run_scheduler():
    """Runs the background scheduler."""
    schedule.every(10).seconds.do(find_and_classify_pending_cases)
    print("⏰ Classification scheduler started. Checking every 10 seconds.")
    while True:
        schedule.run_pending()
        time.sleep(1)

# ----------------------
# FLASK APP SETUP
# ----------------------
app = Flask(__name__)
CORS(app)


# --- 7. FLASK ROUTES ---
@app.route("/api/upload/<userid>/<status>", methods=["POST"])
def handle_upload(userid, status):
    print(f"\n---> STARTING UPLOAD attempt for user={userid}")
    
    try:
        s3_resource = get_s3_resource()
        s3_client = get_s3_client()
        bucket = s3_resource.Bucket(S3_BUCKET_NAME)
        mstr_key = "Master/mockdata_with_category_images.json"

        # 1. GENERATE USER-SPECIFIC SEQUENTIAL CASE ID
        caseid = get_next_caseid(userid) 
        
        # 2. CHECK & PARSE METADATA
        metadata_file = request.files.get("jsonFile")
        if not metadata_file:
            return jsonify({"error": "Missing metadata JSON file (Expected key: jsonFile)"}), 400
        
        metadata_content = metadata_file.read().decode("utf-8") 
        metadata_data = json.loads(metadata_content)

        s3_case_path = f"{userid}/{caseid}"
        print(f"INFO: Case ID resolved: {caseid}. Saving metadata...")

        # 3. SAVE METADATA TO S3 (Use the sequential ID)
        metadata_data['caseid'] = caseid
        updated_metadata_content = json.dumps(metadata_data)
        meta_key = f"{s3_case_path}/metadata.json"
        bucket.put_object(
            Key=meta_key, Body=updated_metadata_content.encode("utf-8"), ContentType="application/json"
        )
        print("INFO: Metadata saved successfully.")

        # 4. SAVE MEDIA FILES TO S3
        media_files = request.files.getlist("files")
        if not media_files:
            return jsonify({"error": "No media files uploaded (Expected key: files)"}), 400

        # Get S3 URL prefix for Master record
        location_constraint = s3_client.get_bucket_location(Bucket=S3_BUCKET_NAME)["LocationConstraint"] or AWS_REGION
        
        media_records = []
        for media_file in media_files:
            filename = media_file.filename
            if not filename: continue
                 
            s3_pending_key = f"{s3_case_path}/Pending/{filename}"
            content_type = media_file.content_type if media_file.content_type else 'application/octet-stream'
            
            bucket.put_object(Key=s3_pending_key, Body=media_file.read(), ContentType=content_type)
            
            media_url = f"https://s3-{location_constraint}.amazonaws.com/{S3_BUCKET_NAME}/{s3_pending_key}"
            media_records.append({"url": media_url, "type": content_type.split("/")[0]})
            print(f"✅ Media file saved: {s3_pending_key}")

        # 5. CREATE INITIAL PENDING MASTER RECORD
        date_str = metadata_data.get("date", datetime.datetime.now().isoformat())
        date_object = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ").date()
        closed_date_str = (date_object + datetime.timedelta(days=10)).strftime("%Y-%m-%d")
        
        # Reverse geocoding for location resolution
        latitude = float(metadata_data.get("latitude", 0.0))
        longitude = float(metadata_data.get("longitude", 0.0))
        reported_city = metadata_data.get("city", "").strip() # Original city from frontend metadata

        # If the frontend provided a junk city name, use the Geocoding API
        is_city_valid = (reported_city and reported_city.lower() not in ["n/a", "undefined", "undefined telan"])
        resolved_location = reported_city if is_city_valid else get_location_from_coords(latitude, longitude) 

        new_pending_record = {
            "Date": date_object.strftime("%Y-%m-%d"),
            "User ID": userid, "Case ID": caseid,
            "Location": resolved_location,
            "latitude": latitude, "longitude": longitude,
            "Category": "Pending Classification", # TEMPORARY LABEL
            "Status": "Pending", # TEMPORARY STATUS
            "Priority": "4",
            "ClosedDate": closed_date_str,
            "MediaFiles": media_records,
            "Description": metadata_data.get("description", ""), # ADDED: Saving the user's description
            "Department": "Pending", # Initialize Department field
        }
        
        # Load existing Master data list
        mstr_obj = bucket.Object(mstr_key)
        try:
            mstr_data = json.load(mstr_obj.get()["Body"])
        except:
            mstr_data = []

        mstr_data.append(new_pending_record)
        mstr_obj.put(Body=json.dumps(mstr_data, indent=4).encode("utf-8"))
        print(f"✅ Masterdata APPENDED with initial PENDING record for Case {caseid}. UI ready for fetch.")

        # 6. Trigger Classification (AUTOMATIC)
        threading.Thread(target=classifier, args=(userid, caseid), daemon=True).start()
        print("INFO: Classification thread started.")
        
        return jsonify({
            "message": "Upload complete. Card added to UI (Pending Classification).",
            "userid": userid, "caseid": caseid 
        }), 200

    except Exception as e:
        print("\n" + "="*50)
        print("❌ CRITICAL UNHANDLED 500 ERROR DURING UPLOAD")
        print(traceback.format_exc())
        print("="*50)
        return jsonify({"error": "An internal server error occurred"}), 500


@app.route("/api/Mastercases/Master", methods=["GET"])
def serve_master_data():
    """Serve all master cases."""
    try:
        mstr_obj = get_s3_resource().Object(S3_BUCKET_NAME, "Master/mockdata_with_category_images.json")
        data = json.load(mstr_obj.get()["Body"])
        return jsonify({"data": data, "message": "Master cases fetched successfully"}), 200
    except Exception as e:
        print(f"❌ Error serving master data: {e}")
        return jsonify({"error": "Failed to retrieve master data", "details": str(e)}), 500

@app.route("/api/cases-with-images-json/<userid>", methods=["GET"])
def get_cases_for_user(userid):
    """Fetch and format cases for a specific user."""
    try:
        mstr_obj = get_s3_resource().Object(S3_BUCKET_NAME, "Master/mockdata_with_category_images.json")
        data = json.load(mstr_obj.get()["Body"])

        user_cases = [case for case in data if str(case.get("User ID")) == str(userid)]

        # --- TRANSFORM MASTER RECORDS TO FRONTEND FORMAT ---
        formatted_cases = []
        for case in user_cases:
            resolved_department = case.get("Department", case.get("Category", "N/A"))
            
            formatted_cases.append({
                "caseid": case.get("Case ID"),
                "images": [
                    {"fileUrl": m.get("url"), "type": m.get("type")}
                    for m in case.get("MediaFiles", [])
                ],
                "jsonFiles": {
                    "metadata": {
                        "date": case.get("Date"),
                        "city": case.get("Location"), 
                        "latitude": case.get("latitude"),
                        "longitude": case.get("longitude"),
                        "description": case.get("Description"), 
                    },
                    "result": {
                        "label": case.get("Category"),
                        "Priority": case.get("Priority"),
                        "reason": "", 
                        "department": resolved_department, 
                    }
                },
                "status": case.get("Status", "Unknown")
            })

        return jsonify({
            "userid": userid,
            "cases": formatted_cases,
            "message": "User cases fetched successfully"
        }), 200

    except Exception as e:
        print(f"❌ Error fetching user cases: {e}")
        return jsonify({"error": "Failed to fetch user cases", "details": str(e)}), 500

@app.route('/')
def home():
    return jsonify({"message": "Retizen AI Flask Backend Running"}), 200

@app.route('/service-worker.js')
def service_worker():
    return "", 204

# ---------------- AWS LAMBDA HANDLER ----------------
def handler(event, context):
    try:
        user_id = event["queryStringParameters"]["userid"]
        case_id = event["queryStringParameters"]["caseid"]
        result = classifier(user_id, case_id)
        return {"statusCode": 200, "body": result}
    except Exception as e:
        return {"statusCode": 500, "body": str(e)}

# --- 8. MAIN DRIVER ---
if __name__ == "__main__":
    # Start the background scheduler in its own thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()

    print("🚀 Retizen Flask backend running at http://127.0.0.1:3001/")

    app.run(host='0.0.0.0', port=3001, debug=True, use_reloader=False)
