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
import schedule
import requests # <-- UNCOMMENTED: Used for Google Maps Geocoding
import os
# ---------------- CONFIGURATION ----------------

# ----------------------------------------------
GEMINI_API_KEY = "AIzaSyBfvJVOK9idpF-c0q1TS1-jlUn4uyyhR8w" # NOTE: Using the key you provided for Gemini



GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
GEOCODE_API_KEY = os.getenv("GEOCODE_API_KEY")

# ---------------- FLASK APP SETUP ----------------
app = Flask(__name__)
CORS(app)


# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name="gemini-2.5-pro")

# ---------------- PROMPT ----------------

description = f"""
Analyze the content provided (media and accompanying text description) and determine the most appropriate category. The content may represent an incident, behavior, issue, or general public concern. Use the provided guidelines to evaluate the content and classify it accordingly.
"""

# Define the classification prompt
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


# ---------------- DICTIONARIES ----------------
mimetype_dict = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "heic": "image/heic",
    "mp3": "audio/mpeg",
}

department_dict = {
    "Public Safety": "Public Safety",
    "Law Enforcement": "Legal",
    "Behavioral": "Crime",
    "Junk": "None",
    "Unknown": "Unknown",
}

# ---------------- SAFE GEMINI CALL ----------------
def safe_generate(model, prompt):
    retries = 3
    for attempt in range(retries):
        try:
            return model.generate_content(prompt)
        except Exception as e:
            print(f"⚠️ Gemini error: {e}")
            if attempt < retries - 1:
                delay = 5 * (attempt + 1)
                print(f"⏳ Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise

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

# ---------------- GEOCoding FUNCTION (UPDATED) ----------------
def get_location_from_coords(latitude, longitude):
    """
    Reverse geocodes the coordinates using the Google Maps Geocoding API.
    """
    if latitude == 0.0 and longitude == 0.0:
        return "N/A (No Coords)"

    print(f"🗺️ Attempting to geocode coords: {latitude}, {longitude}...")
    
    try:
        url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={latitude},{longitude}&key={GEOCODE_API_KEY}"
        response = requests.get(url).json()
        
        if response.get("status") == "OK" and response.get("results"):
            # Attempt to find locality/city in the results
            for result in response["results"]:
                for component in result["address_components"]:
                    # Priority 1: City/Locality
                    if "locality" in component["types"]:
                        return component["long_name"]
                    # Priority 2: Postal Town / Administrative area 2 (County/District)
                    if "postal_town" in component["types"] or "administrative_area_level_2" in component["types"]:
                        return component["long_name"]

            # Fallback to the formatted address if no city/town is found
            return response["results"][0]["formatted_address"]
        elif response.get("status") == "ZERO_RESULTS":
            return f"No location found for {latitude}, {longitude}"
        else:
            print(f"❌ Geocoding API failed. Status: {response.get('status')}")
            return f"Coords: {latitude}, {longitude} (API Error)"

    except requests.exceptions.RequestException as e:
        print(f"❌ Geocoding Request failed: {e}")
        return f"Coords: {latitude}, {longitude} (Request Error)"

# ---------------- GEMINI CLASSIFIER ----------------
def score(file_stream, mimetype):
    """Uploads media to Gemini and returns classification."""
    # Reset file stream pointer to the beginning before uploading
    file_stream.seek(0)
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
        "label": "Junk",
        "reason": "Setup error",
        "Priority": "4",
    }

    try:
        s3 = get_s3_resource()
        bucket = s3.Bucket(S3_BUCKET_NAME)
        
        # Check if result.json already exists (case already classified)
        result_key = f"{userid}/{caseid}/Result/result.json"
        try:
            s3.Object(S3_BUCKET_NAME, result_key).load()
            print(f"⏭️ Case {userid}/{caseid} already classified (result.json exists). Skipping.")
            return json.dumps({"message": "Already classified"})
        except:
            pass

        # Find media file
        file_paths = [
            obj.key
            for obj in bucket.objects.filter(Prefix=s3_dir_path)
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

            # Check if city is junk (undefined, N/A, empty string, etc.)
            is_city_valid = (
                reported_city and 
                reported_city.strip() and 
                reported_city.lower() not in ["n/a", "undefined", "undefined telan"]
            )
            
            if is_city_valid:
                resolved_location = reported_city
            else:
                # If city is missing or junk, use geocoding
                resolved_location = get_location_from_coords(latitude, longitude)
            
            print(f"📍 Resolved location for Masterdata: {resolved_location}")
            # --- END LOCATION RESOLUTION ---
            
            media_url = f"https://s3-{AWS_REGION}.amazonaws.com/{S3_BUCKET_NAME}/{file_path}"
            media_type = obj.content_type.split("/")[0] if obj.content_type else "unknown"

            # Safely parse date
            date_str = meta_data.get("date", datetime.datetime.now().isoformat())
            try:
                date_object = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ").date()
            except ValueError:
                date_object = datetime.date.today()
            
            closed_date_str = (date_object + datetime.timedelta(days=10)).strftime("%Y-%m-%d")

            new_record = {
                "Date": date_object.strftime("%Y-%m-%d"),
                "User ID": userid,
                "Case ID": caseid,
                "Location": resolved_location, # Use the resolved location here
                "latitude": latitude,
                "longitude": longitude,
                "Category": final_response["label"],
                "Status": "Open",
                "Priority": final_response["Priority"],
                "ClosedDate": closed_date_str,
                "MediaFiles": [{"url": media_url, "type": media_type}],
            }

            # Prevent duplicate entries in Masterdata
            if not any(rec.get("Case ID") == caseid and rec.get("User ID") == userid for rec in mstr_data):
                mstr_data.append(new_record)
                mstr_obj.put(Body=json.dumps(mstr_data, indent=4).encode("utf-8"))
                print("✅ Appended classification result to Masterdata.")
            else:
                print("⏭️ Masterdata already contains this case. Skipping Masterdata update.")

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

# ---------------- NEW SCHEDULING LOGIC ----------------

def find_and_classify_pending_cases():
    """Scans S3 for new 'Pending' cases that don't have a 'Result' yet."""
    print("\n🔍 Running scheduled check for pending classifications...")
    try:
        s3 = get_s3_resource()
        bucket = s3.Bucket(S3_BUCKET_NAME)

        # Strategy: Find all unique user/case paths that have a 'Pending' subfolder.
        all_pending_objects = bucket.objects.filter(Prefix="")
        pending_cases = set()

        for obj in all_pending_objects:
            # Example key: anu/9/Pending/file.jpg
            parts = obj.key.split('/')
            if len(parts) >= 3 and parts[2] == 'Pending' and not obj.key.endswith('/'):
                userid = parts[0]
                caseid = parts[1]
                result_key = f"{userid}/{caseid}/Result/result.json"

                # Check if result.json exists. If not, add to pending list.
                try:
                    s3.Object(S3_BUCKET_NAME, result_key).load()
                except:
                    # Only add if it hasn't been added and has a file
                    if (userid, caseid) not in pending_cases:
                        pending_cases.add((userid, caseid))

        if not pending_cases:
            print("No new pending cases found.")
            return

        print(f"Found {len(pending_cases)} case(s) to classify: {pending_cases}")
        
        # Trigger classification for each pending case asynchronously
        for userid, caseid in pending_cases:
            # Start a new thread for each classification to prevent blocking
            threading.Thread(target=classifier, args=(userid, caseid), daemon=True).start()

    except Exception as e:
        print(f"❌ Error in scheduled job: {e}")

def run_scheduler():
    """Runs the background scheduler."""
    # Schedule the function to run every 10 seconds
    schedule.every(10).seconds.do(find_and_classify_pending_cases)
    
    print("⏰ Classification scheduler started. Checking every 10 seconds.")
    while True:
        schedule.run_pending()
        time.sleep(1)

# ---------------- FLASK ROUTES ----------------

@app.route("/api/upload_complete", methods=["POST"])
def upload_complete():
    """🚀 Auto-trigger classification after upload."""
    try:
        data = request.get_json()
        userid = data.get("userid")
        caseid = data.get("caseid")

        if not userid or not caseid:
            return jsonify({"error": "Missing userid or caseid"}), 400

        print(f"📩 Upload complete signal received for user={userid}, case={caseid}")

        # Run classification asynchronously in background immediately upon trigger
        threading.Thread(target=classifier, args=(userid, caseid), daemon=True).start()

        return jsonify({
            "message": "Upload received. Auto-classification running in background."
        }), 200

    except Exception as e:
        print(f"❌ Error in /api/upload_complete: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/Mastercases/Master", methods=["GET"])
def serve_master_data():
    """Serve all master cases for frontend cards."""
    try:
        s3 = get_s3_resource()
        mstr_obj = s3.Object(S3_BUCKET_NAME, "Master/mockdata_with_category_images.json")
        data = json.load(mstr_obj.get()["Body"])
        return jsonify({"data": data, "message": "Master cases fetched successfully"}), 200
    except Exception as e:
        print(f"❌ Error serving master data: {e}")
        return jsonify({"error": "Failed to retrieve master data", "details": str(e)}), 500

@app.route('/')
def home():
    return jsonify({"message": "Retizen AI Flask Backend Running"}), 200

@app.route('/service-worker.js')
def service_worker():
    # Dummy route to silence browser 404 warnings
    return "", 204


@app.route("/api/cases-with-images-json/<userid>", methods=["GET"])
@app.route("/api/cases-with-images-json/<userid>", methods=["GET"])
def get_cases_for_user(userid):
    try:
        s3 = get_s3_resource()
        mstr_obj = s3.Object(S3_BUCKET_NAME, "Master/mockdata_with_category_images.json")
        data = json.load(mstr_obj.get()["Body"])

        user_cases = [case for case in data if str(case.get("User ID")) == str(userid)]

        # --- TRANSFORM MASTER RECORDS TO FRONTEND FORMAT ---
        formatted_cases = []
        for case in user_cases:
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
                        "longitude": case.get("longitude")
                    },
                    "result": {
                        "label": case.get("Category"),
                        "Priority": case.get("Priority"),
                        "reason": "",
                        "department": case.get("Category"),
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


# ---------------- AWS LAMBDA HANDLER ----------------
def handler(event, context):
    try:
        user_id = event["queryStringParameters"]["userid"]
        case_id = event["queryStringParameters"]["caseid"]
        print(f"Lambda triggered for user={user_id}, case={case_id}")
        result = classifier(user_id, case_id)
        return {"statusCode": 200, "body": result}
    except Exception as e:
        print(f"Lambda error: {e}")
        return {"statusCode": 500, "body": str(e)}

# ---------------- MAIN DRIVER ----------------
if __name__ == "__main__":
    # Start the background scheduler in its own thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()

    print("🚀 Retizen Flask backend running at http://127.0.0.1:3001/")

    app.run(port=3001, debug=True, use_reloader=False)






