import time
import argparse
import random
import csv
import os
import json 
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime,timedelta

import requests

from fastembed import TextEmbedding
import numpy as np


# Initialize fastembed model
EMBED_MODEL = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

try:
    from sense_hat import SenseHat
    sense = SenseHat()
    sense.clear()
    SENSE_AVAILABLE = True
except ImportError:
    sense = None
    SENSE_AVAILABLE = False

def show_correct_symbol():
    """Show GREEN check mark on Sense HAT"""
    if not SENSE_AVAILABLE:
        print("[LED] Correct symbol (✔)")
        return

    G = (0, 255, 0)   # Green
    O = (0, 0, 0)     # Off

    check = [
        O,O,O,O,O,O,O,O,
        O,O,O,O,O,O,G,O,
        O,O,O,O,O,G,O,O,
        G,O,O,O,G,O,O,O,
        O,G,O,G,O,O,O,O,
        O,O,G,O,O,O,O,O,
        O,O,O,O,O,O,O,O,
        O,O,O,O,O,O,O,O
    ]

    sense.set_pixels(check)
    time.sleep(1)
    sense.clear()


def show_wrong_symbol():
    """Show RED cross on Sense HAT"""
    if not SENSE_AVAILABLE:
        print("[LED] Wrong symbol (✖)")
        return

    R = (255, 0, 0)   # Red
    O = (0, 0, 0)     # Off

    cross = [
        R,O,O,O,O,O,O,R,
        O,R,O,O,O,O,R,O,
        O,O,R,O,O,R,O,O,
        O,O,O,R,R,O,O,O,
        O,O,O,R,R,O,O,O,
        O,O,R,O,O,R,O,O,
        O,R,O,O,O,O,R,O,
        R,O,O,O,O,O,O,R
    ]

    sense.set_pixels(cross)
    time.sleep(1)
    sense.clear()
    
def show_temperature_idle():
    """
    Display current temperature on Sense HAT LED matrix when app is idle.
    """
    if not SENSE_AVAILABLE:
        return

    temp = sense.get_temperature()
    temp = round(temp, 1)

    sense.show_message(
        f"{temp}C",
        scroll_speed=0.08,
        text_colour=(0, 0, 255)
    )

# ===== Status Symbols =====

B = (0, 0, 255)   # Blue
G = (0, 255, 0)   # Green
R = (255, 0, 0)   # Red
O = (0, 0, 0)     # Off

START_SYMBOL = [
 O,O,O,G,G,O,O,O,
 O,G,O,G,G,O,G,O,
 G,O,O,G,G,O,O,G,
 G,O,O,G,G,O,O,G,
 G,O,O,G,G,O,O,G,
 G,O,O,O,O,O,O,G,
 O,G,O,O,O,O,G,O,
 O,O,G,G,G,G,O,O
]

GAME_START_SYMBOL = [
 O,O,O,O,G,O,O,O,
 O,O,O,O,G,G,O,O,
 O,O,O,O,G,G,G,O,
 O,O,O,O,G,G,G,G,
 O,O,O,O,G,G,G,O,
 O,O,O,O,G,G,O,O,
 O,O,O,O,G,O,O,O,
 O,O,O,O,O,O,O,O
]

GAME_EXIT_SYMBOL = [
 O,O,O,O,O,O,O,O,
 O,R,R,R,R,R,R,O,
 O,R,R,R,R,R,R,O,
 O,R,R,R,R,R,R,O,
 O,R,R,R,R,R,R,O,
 O,R,R,R,R,R,R,O,
 O,R,R,R,R,R,R,O,
 O,O,O,O,O,O,O,O
]


def show_symbol(symbol, delay=1.5):
    if not SENSE_AVAILABLE:
        return
    sense.set_pixels(symbol)
    time.sleep(delay)
    sense.clear()

def show_score_on_led(score):
    if not SENSE_AVAILABLE:
        return
    sense.show_message(
        f"Score:{score}",
        scroll_speed=0.08,
        text_colour=(255, 255, 0)
    )

######################### Weather ########################
WEATHER_API_KEY = "14092eaa1f7e9716920780eb8684ecbb"
DEFAULT_CITY = "Wolfenbuettel"  # ASCII-safe for OpenWeather

# ===============================
# Semantic Location Intents
# ===============================
LOCATION_INTENT_LABELS = [
    "where is a lecture hall",
    "location of a university building",
    "campus building location",
    "university event location",
    "exam location",
    "library location",
    "cafeteria location"
]
LOCATION_INTENT_EMBEDDINGS = np.array([
    next(EMBED_MODEL.embed([label]))
    for label in LOCATION_INTENT_LABELS
])
def is_location_question_semantic(question: str, threshold=0.55):
    q_vec = next(EMBED_MODEL.embed([question.lower()]))

    scores = LOCATION_INTENT_EMBEDDINGS @ q_vec / (
        np.linalg.norm(LOCATION_INTENT_EMBEDDINGS, axis=1) *
        np.linalg.norm(q_vec)
    )

    return np.max(scores) >= threshold

def fetch_weather(city: str):
    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={WEATHER_API_KEY}&units=metric"
    )

    try:
        r = requests.get(url, timeout=5)
        data = r.json()

        if r.status_code != 200:
            return " Weather data currently unavailable."

        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]

        return f"Weather in {city}: {temp}°C, {desc}"

    except Exception:
        return "Weather information unavailable."


TEMP_LOG_FILE = "temperature_log.json"
MONITOR_INTERVAL_SECONDS = 1800  # ~30 minutes
TEMP_RAW_LOG_FILE = "temperature_raw_log.json"
def log_raw_temperature():
    """
    Store raw temperature readings every ~30 minutes with timestamp.
    """
    if not SENSE_AVAILABLE:
        return

    now = datetime.now()
    entry = {
        "timestamp": now.timestamp(),
        "time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "temp": round(sense.get_temperature(), 2)
    }

    data = []
    if os.path.exists(TEMP_RAW_LOG_FILE):
        with open(TEMP_RAW_LOG_FILE, "r") as f:
            data = json.load(f)

    data.append(entry)

    with open(TEMP_RAW_LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)
LECTURE_START_HOUR = 9
LECTURE_END_HOUR = 17


def get_average_temperature_for_location(days=3):
    """
    Calculate average temperature for lecture/event period
    over the last N days.
    """
    if not os.path.exists(TEMP_RAW_LOG_FILE):
        return "No temperature history available."

    with open(TEMP_RAW_LOG_FILE, "r") as f:
        data = json.load(f)

    cutoff = datetime.now().timestamp() - (days * 24 * 3600)
    temps = []

    for entry in data:
        ts = entry["timestamp"]
        t = datetime.fromtimestamp(ts)

        if ts >= cutoff and LECTURE_START_HOUR <= t.hour < LECTURE_END_HOUR:
            temps.append(entry["temp"])

    if not temps:
        return "Not enough temperature data for the lecture period."

    avg_temp = round(sum(temps) / len(temps), 2)
    return f"Average temperature during lecture period (last {days} days): {avg_temp}°C"



def log_local_temperature():
    """
    Monitor local temperature every ~30 minutes and update daily min/max.
    """
    if not SENSE_AVAILABLE:
        return

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    temp = round(sense.get_temperature(), 2)

    data = {}

    if os.path.exists(TEMP_LOG_FILE):
        with open(TEMP_LOG_FILE, "r") as f:
            data = json.load(f)

    if today not in data:
        data[today] = {
            "min": temp,
            "max": temp,
            "last_updated": now.timestamp()
        }
    else:
        data[today]["min"] = min(data[today]["min"], temp)
        data[today]["max"] = max(data[today]["max"], temp)
        data[today]["last_updated"] = now.timestamp()

    with open(TEMP_LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_weather_min_max(city: str):
    """
    Get today's min/max temperature from weather forecast API.
    """
    url = (
        "https://api.openweathermap.org/data/2.5/forecast"
        f"?q={city}&appid={WEATHER_API_KEY}&units=metric"
    )

    try:
        r = requests.get(url, timeout=5)
        data = r.json()

        temps = [
            item["main"]["temp"]
            for item in data["list"]
            if datetime.fromtimestamp(item["dt"]).date() == datetime.now().date()
        ]

        if not temps:
            return None

        return round(min(temps), 2), round(max(temps), 2)

    except Exception:
        return None


def get_last_3_days_temperature_change():
    """
    Return temperature increase/decrease (max-min) per day
    for the last 3 days for both local sensor and weather forecast.
    """
    if not os.path.exists(TEMP_LOG_FILE):
        return "No local temperature data available."

    with open(TEMP_LOG_FILE, "r") as f:
        local_data = json.load(f)

    today = datetime.now().date()
    output = []

    for i in range(3):
        day = today - timedelta(days=i)
        day_str = day.strftime("%Y-%m-%d")

        if day_str not in local_data:
            continue

        local_min = local_data[day_str]["min"]
        local_max = local_data[day_str]["max"]
        local_diff = round(local_max - local_min, 2)

        forecast = get_weather_min_max(DEFAULT_CITY)
        if forecast:
            w_min, w_max = forecast
            weather_diff = round(w_max - w_min, 2)
        else:
            weather_diff = "N/A"

        output.append(
            f"{day_str} | "
            f"Local ΔT: {local_diff}°C | "
            f"Forecast ΔT: {weather_diff}°C"
        )

    if not output:
        return "Not enough temperature data collected yet."

    return "\n".join(output)


def temperature_monitor_loop():
    """
    Continuous monitoring loop for Raspberry Pi.
    """
    try:
        while True:
            log_local_temperature()
            time.sleep(MONITOR_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print(f"{current_time()} Temperature monitoring stopped.")





QA_DATA = {
    "what is your name?": [
        "I'm TerminalTalk, your terminal assistant!",
        "You can call me TerminalTalk.",
        "I'm TerminalTalk, here to help you in the terminal."
    ],
    "what is a variable?": [
        "A variable is a named location in memory used to store data for your program.",
        "A variable stores a value in memory so your program can use it later."
    ],
    "what is a function?": [
        "A function is a block of reusable code that performs a specific task.",
        "A function groups code so you can reuse it by calling its name."
    ],
    "what is oop?": [
        "OOP stands for Object-Oriented Programming — a way to structure code using classes and objects.",
        "OOP (Object-Oriented Programming) organizes code around objects and classes."
    ],
    "what is git?": [
        "Git is a version control system used to track changes in your code.",
        "Git helps you track, manage, and collaborate on changes in your codebase."
    ],
    "what is api?": [
        "API stands for Application Programming Interface — a way for programs to communicate with each other.",
        "An API lets different software components talk to each other using defined rules."
    ],
    "what is debugging?": [
        "Debugging is the process of finding and fixing errors in your code.",
        "Debugging means locating and fixing bugs in your program."
    ],
    "what is machine learning?": [
        "Machine learning is a type of AI that allows systems to learn from data and improve over time.",
        "Machine learning lets computers learn patterns from data instead of being explicitly programmed."
    ],
    "what is json?": [
        "JSON stands for JavaScript Object Notation — a lightweight format for storing and transferring data.",
        "JSON is a text format for representing structured data, often used in APIs."
    ],
    "what is a loop?": [
        "A loop is a control structure that repeats a block of code multiple times.",
        "A loop runs the same code again and again until a condition is met."
    ],
    "what is recursion?": [
        "Recursion is when a function calls itself to solve smaller instances of a problem.",
        "Recursion means solving a problem by breaking it into smaller subproblems and calling the same function."
    ],
    "what is the deadline for paying the semester fee?": [
        "The deadline for paying the semester fee is usually a few weeks before the semester starts. Please check your student portal for the exact deadline.",
        "You must pay the semester fee before the re-registration deadline. The exact date is shown in your enrollment portal."
    ],
    "how can i register for semester 2?": [
        "You can register for semester 2 through the university's online student portal under the re-registration section.",
        "To register for semester 2, log in to your student account and follow the re-registration instructions provided there.",
        "Registration is done through the student portal using your university login."
    ],
    "where can i find the exam schedule?": [
        "You can find the exam schedule on the university website or in your student portal, typically under 'Examination dates'."
    ],
    "where can i find lecture hall 203?": [
        "Lecture hall 203 is located in the main campus building. Please follow the campus signage or check the campus map near the entrance.",
        "You can find lecture hall 203 by looking at the campus map or using the room search on the university website."
    ],
   
    "how can i get to building a?": [
        "Building A is near the main entrance of the campus. Follow the signs or check the campus map for the walking route.",
        "To get to Building A, enter the campus from the main gate and follow the signs labelled 'Building A'."
    ],
    "where can i find the library?": [
        "The library is usually located in the central campus area. Check the campus map or follow the signs to 'Library'.",
        "You can find the library by following the direction signs on campus or by checking the building list on the university website."
    ],
    "what are the library opening hours?": [
        "Library opening hours vary by semester. Please check the library section on the university website for the current schedule.",
        "The library posts its opening hours online and at the entrance. Make sure to check there for the latest information."
    ],
    "where can i find the cafeteria?": [
        "The cafeteria is typically located near the main building or student center. Follow the signs to 'Mensa' or 'Cafeteria'.",
        "You can find the cafeteria by checking the campus map or asking at the information desk at the main entrance."
    ],
    "where can i get my student id card?": [
        "You can collect your student ID card from the admissions office.",
        "You can pick up your student ID card at the university’s enrollment or student services desk.",
        "Student ID cards are issued by the admissions department—just bring a valid photo ID."
    ],"when does the semester start?": [
        "The semester usually starts at the beginning of October. Please check the academic calendar for exact dates.",
        "You can find the official semester start date in your university’s academic schedule."
    ],
    "when does the semester end?": [
        "The semester typically ends in February or July depending on the term.",
        "Please check the university academic calendar for the exact end date."
    ],
    "how can i register for courses?": [
        "You can register for courses through the online student portal.",
        "Course registration is done in the student portal under 'Course Management'."
    ],
    "how do i access the student portal?": [
        "You can access the student portal via the university website using your login credentials.",
        "Go to the university homepage and click on 'Student Portal' to log in."
    ],
    "where can i get my timetable?": [
        "You can download your timetable from the student portal.",
        "Timetables are available under 'My Courses' in the student portal."
    ],
    "how can i reset my university password?": [
        "You can reset your password using the 'Forgot Password' option on the login page.",
        "Contact the IT Helpdesk if you are unable to reset your password online."
    ],
    "where is the examination office?": [
        "The examination office is located in Building B, first floor.",
        "You can find the exam office in Building B. Follow the signs at the entrance."
    ],
    "how can i contact the examination office?": [
        "You can reach the examination office via email or through your student portal.",
        "Visit the exam office webpage for contact details and opening hours."
    ],
    "where can i see my exam results?": [
        "Exam results are published in the student portal under 'Exams'.",
        "You can check your exam results online by logging into your student account."
    ],
    "how do i register for exams?": [
        "You can register for exams through the student portal under 'Exams'.",
        "Exam registration is done online. Check the deadlines in your portal."
    ],
}

TRIVIA_QUESTIONS = [
    {
        "question": "What is the name of the University's online student portal?",
        "options": ["A. Moodle", "B. UniPortal", "C. CampusNet", "D. LearnSpace"],
        "correct": "C"
    },
    {
        "question": "Where can students find the official exam schedule?",
        "options": ["A. Cafeteria menu board", "B. Student portal", "C. Gym notice board", "D. Library basement"],
        "correct": "B"
    },
    {
        "question": "What document is usually required to collect a student ID card?",
        "options": ["A. Passport photo", "B. Valid photo ID", "C. Doctor's note", "D. Birth certificate"],
        "correct": "B"
    },
    {
        "question": "Which building contains the main library?",
        "options": ["A. Building A", "B. Building L", "C. Building C", "D. Building B"],
        "correct": "A"
    },
    {
        "question": "Where can students reset their university password?",
        "options": ["A. The examination office", "B. The IT helpdesk portal", "C. The gym registration desk", "D. Lecture hall 203"],
        "correct": "B"
    },
    {
        "question": "What does OOP stand for in computer science?",
        "options": ["A. Office Operation Plan", "B. Object-Oriented Programming", "C. Official Online Process", "D. Online Output Page"],
        "correct": "B"
    },
    {
        "question": "Which campus location is typically used for large exams?",
        "options": ["A. The gym hall", "B. Lecture hall 203", "C. Building A rooftop", "D. The library study rooms"],
        "correct": "B"
    },
    {
        "question": "Where can a student get their semester timetable?",
        "options": ["A. Student portal", "B. Library desk", "C. Campus café", "D. By email from professors"],
        "correct": "A"
    },
    {
        "question": "Which service does the cafeteria primarily provide?",
        "options": ["A. Free textbooks", "B. Meals and drinks", "C. Parking permits", "D. Medical check-ups"],
        "correct": "B"
    },
    {
        "question": "Who can help students with academic registration issues?",
        "options": ["A. IT department", "B. Admissions office", "C. Sports center staff", "D. Cafeteria chef"],
        "correct": "B"
    },
    {
        "question": "What is JSON commonly used for?",
        "options": ["A. Drawing images", "B. Storing structured data", "C. Writing essays", "D. Software installation"],
        "correct": "B"
    },
    {
        "question": "Which office manages exam results?",
        "options": ["A. Admissions office", "B. Examination office", "C. Library helpdesk", "D. Campus security"],
        "correct": "B"
    },
    {
        "question": "Where does the semester typically start?",
        "options": ["A. Beginning of October", "B. End of December", "C. Middle of June", "D. March 1st"],
        "correct": "A"
    },
    {
        "question": "What programming structure allows repeating actions?",
        "options": ["A. Loop", "B. Folder", "C. Filter", "D. Module"],
        "correct": "A"
    },
    {
        "question": "What do students usually use the library for?",
        "options": ["A. Sleeping", "B. Studying and borrowing books", "C. Printing ID cards", "D. Submitting exam papers"],
        "correct": "B"
    }
]


QUESTION_LIST = None
QUESTION_EMBEDDINGS = None

""" Return the currently active questions:
    - If a CSV was imported, use IMPORTED_QA.
    - Otherwise, use INTERNAL_QA.
"""
def known_questions():
    global IMPORTED_QA, QA_DATA

    if IMPORTED_QA is not None:
        return IMPORTED_QA
    
    return QA_DATA

def rebuild_embeddings():
    """Rebuild embeddings for all known questions using fastembed."""
    global QUESTION_LIST, QUESTION_EMBEDDINGS

    qa = known_questions()
    QUESTION_LIST = list(qa.keys())

    # Generate embeddings
    QUESTION_EMBEDDINGS = np.array([
        next(EMBED_MODEL.embed([q])) for q in QUESTION_LIST
    ])


IMPORTED_QA = None
rebuild_embeddings()



def current_time():
    return time.strftime("%H:%M:%S")


def list_questions():
    qa_dict = known_questions()
    print(f"{current_time()} Listing {len(qa_dict)} questions:\n")
    for q in qa_dict.keys():
        print(f"- {q}")


def safe_import_csv(filepath: str):
    try:
        logger = logging.getLogger(__name__)

        logger.info("Attempting CSV import: %s", filepath)
        # Check file exists
        if not os.path.exists(filepath):
            logger.warning("CSV import failed: file not found: %s", filepath)
            print(f"{current_time()} ERROR: File path not found: {filepath}")
            print(f"{current_time()} Import aborted. Using internal questions.")
            return None

        # Check extension
        if not filepath.lower().endswith(".csv"):
            print(f"{current_time()} ERROR: Unsupported file type. Only .csv allowed.")
            print(f"{current_time()} Import aborted. Using internal questions.")
            return None

        # Try to open file to check access rights
        with open(filepath, "r", encoding="utf-8") as f:
            pass  # we only check whether opening is possible

        # If everything is okay → call the real CSV loader
        load_questions_from_csv(filepath)

    except PermissionError:
        logger.warning("CSV import permission error for file: %s", filepath)
        print(f"{current_time()} ERROR: Insufficient access rights to read file.")
        print(f"{current_time()} Import aborted. Using internal questions.")
        return None

    except UnicodeDecodeError:
        logger.warning("CSV import corrupted or badly encoded: %s", filepath)
        print(f"{current_time()} ERROR: CSV file appears corrupted or has invalid encoding.")
        print(f"{current_time()} Import aborted. Using internal questions.")
        return None

    except csv.Error:
        logger.warning("CSV file format is invalid/corrupted.: %s", filepath)

        print(f"{current_time()} ERROR: CSV file format is invalid/corrupted.")
        print(f"{current_time()} Import aborted. Using internal questions.")
        return None

    except Exception as e:
        print(f"{current_time()} Unexpected import error: {e}")
        print(f"{current_time()} Import aborted. Using internal questions.")
        return None


def load_questions_from_csv(filepath: str):
    logger = logging.getLogger(__name__)
    logger.info("Loading CSV file: %s", filepath)
    global IMPORTED_QA

    if not os.path.exists(filepath):
        print(f"{current_time()} ERROR: File not found: {filepath}")
        print(f"{current_time()} Falling back to internal questions.")
        IMPORTED_QA = None
        return

    qa_dict = {}
    try:
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames or "question" not in reader.fieldnames:
                print(f"{current_time()} ERROR: CSV must contain at least a 'question' column.")
                print(f"{current_time()} Falling back to internal questions.")
                IMPORTED_QA = None
                return

            for row in reader:
                question = (row.get("question") or "").strip().lower()
                if not question:
                    continue  # skip empty

                answers = []
                for col in ["answer1", "answer2", "answer3", "answer4"]:
                    ans = row.get(col)
                    if ans:
                        ans = ans.strip()
                        if ans:
                            answers.append(ans)

                if not answers:
                    continue

                qa_dict[question] = answers

        if not qa_dict:
            print(f"{current_time()} ERROR: No valid Q&A rows found in CSV.")
            print(f"{current_time()} Falling back to internal questions.")
            IMPORTED_QA = None
        else:
            if len(qa_dict) < 10:
                print(f"{current_time()} WARNING: CSV has only {len(qa_dict)} Q&A pairs (min 10 suggested).")
            IMPORTED_QA = qa_dict
            logger.info("CSV loaded: %d questions imported", len(qa_dict))

            print(f"{current_time()} Imported {len(qa_dict)} questions from CSV file: {filepath}")

    except Exception as e:
        print(f"{current_time()} ERROR while reading CSV: {e}")
        print(f"{current_time()} Falling back to internal questions.")
        IMPORTED_QA = None




def save_qa_to_source(new_qa_dict):
    """
    add a question+answers into INTERNAL_QA inside the .py file.
    """
    global QA_DATA
    QA_DATA = new_qa_dict  # update in-memory dict too

    import_path = os.path.abspath(__file__)

    with open(import_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Find the QA_DATA block start
    start_idx = source.find("QA_DATA =")
    if start_idx == -1:
        print(f"{current_time()} ERROR: 'QA_DATA =' not found in source file.")
        return

    # Find the next 'def current_time' (start of next section)
    end_idx = source.find("def current_time", start_idx)
    if end_idx == -1:
        print(f"{current_time()} ERROR: 'def current_time' not found after QA_DATA.")
        return

    # Convert dict to pretty text
    qa_text = json.dumps(new_qa_dict, indent=4, ensure_ascii=False)

    new_block = f"QA_DATA = {qa_text}\n\n"

    before = source[:start_idx]
    after = source[end_idx:]  # keep the rest of the file

    new_source = before + new_block + after

    with open(import_path, "w", encoding="utf-8") as f:
        f.write(new_source)

    print(f"{current_time()} Internal QA list updated in {os.path.basename(import_path)}.")


def add_question_cli(question: str, answers: list[str]):
    """
    Add a new question OR add one/more answers to an existing question.
    - If the question does not exist: create it with the given answers.
    - If the question exists: append only the new answers.
    """
    q_key = question.strip().lower()
    if not q_key:
        print(f"{current_time()} ERROR: Question text is empty.")
        return

    if not answers:
        print(f"{current_time()} ERROR: At least one --answer is required when using --add.")
        return

    # Start from current QA_DATA
    new_qa = dict(QA_DATA)

    # Ensure we have a list for this question
    existing_answers = list(new_qa.get(q_key, []))

    # Add new answers (avoid duplicates, strip whitespace)
    added_any = False
    for ans in answers:
        ans_clean = ans.strip()
        if not ans_clean:
            continue
        if ans_clean not in existing_answers:
            existing_answers.append(ans_clean)
            added_any = True

    if not existing_answers:
        print(f"{current_time()} ERROR: No valid answers provided.")
        return

    if q_key in new_qa:
        if added_any:
            print(f"{current_time()} Added {len(answers)} answer(s) to existing question: {q_key}")
        else:
            print(f"{current_time()} No new answers added (all duplicates) for question: {q_key}")
    else:
        print(f"{current_time()} Created new question: {q_key}")

    new_qa[q_key] = existing_answers
    save_qa_to_source(new_qa)
    rebuild_embeddings()



def remove_question_cli(question: str, answers: list[str] | None = None):
    """
    Remove a whole question OR specific answers for a question.
    - If 'answers' is None/empty: remove the whole question.
    - If 'answers' is provided: remove only those answers from the list.
    """
    q_key = question.strip().lower()
    if not q_key:
        print(f"{current_time()} ERROR: Question text is empty.")
        return

    new_qa = dict(QA_DATA)

    if q_key not in new_qa:
        print(f"{current_time()} Question not found: {q_key}")
        return

    # If no answers given → remove whole question
    if not answers:
        del new_qa[q_key]
        print(f"{current_time()} Removed entire question: {q_key}")
        save_qa_to_source(new_qa)
        return

    # Remove specific answers
    existing_answers = list(new_qa[q_key])
    original_count = len(existing_answers)

    # Remove all matches
    for ans in answers:
        ans_clean = ans.strip()
        existing_answers = [a for a in existing_answers if a != ans_clean]

    removed_count = original_count - len(existing_answers)

    if removed_count == 0:
        print(f"{current_time()} No matching answers found to remove for question: {q_key}")
        return

    if not existing_answers:
        # No answers left → optionally remove whole question
        print(f"{current_time()} All answers removed; deleting question: {q_key}")
        del new_qa[q_key]
    else:
        new_qa[q_key] = existing_answers
        print(f"{current_time()} Removed {removed_count} answer(s) from question: {q_key}")

    save_qa_to_source(new_qa)
    rebuild_embeddings()



SIMILARITY_THRESHOLD = 0.45  # tune as needed (0.5–0.6 is good)

def get_answer(question):
    logger = logging.getLogger(__name__)
    global QUESTION_LIST, QUESTION_EMBEDDINGS
    logger.debug("get_answer() called with: %s", question)
    logger.info("Received question: %s", question)

    q = question.lower().strip()
    qa_dict = known_questions()

    # 1) Exact or substring match (old behavior preserved)
    for key, answers in qa_dict.items():
        if key == q or key in q or q in key:
            logger.debug("Exact/substring match found. Key='%s', Selected Answer='%s'", key, random.choice(answers))
            # return random.choice(answers)
            base_answer = random.choice(answers)

            if is_location_question_semantic(question):
                weather = fetch_weather(DEFAULT_CITY)
                avg_temp = get_average_temperature_for_location(3)
                return f"{base_answer}\n\n{weather} \n {avg_temp}"

            return base_answer


    # 2) Semantic fastembed match
    if QUESTION_EMBEDDINGS is None:
        logger.debug("Embeddings not loaded. Rebuilding embeddings.")
        rebuild_embeddings()

    # Embed user question
    user_vec = next(EMBED_MODEL.embed([q]))

    # Compute cosine similarity
    scores = QUESTION_EMBEDDINGS @ user_vec / (
        np.linalg.norm(QUESTION_EMBEDDINGS, axis=1) * np.linalg.norm(user_vec)
    )

    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_question = QUESTION_LIST[best_idx]

    logger.debug("Semantic search: Best match='%s', Score=%.3f", best_question, best_score)

    if best_score >= SIMILARITY_THRESHOLD:
        logger.debug("Semantic match accepted. Selected Answer='%s'", random.choice(qa_dict[best_question]))
        logger.info("Semantic match: '%s' -> '%s' (score=%.3f)", question, best_question, best_score)
        # return random.choice(qa_dict[best_question])
        base_answer = random.choice(qa_dict[best_question])

    if is_location_question_semantic(question):
        weather = fetch_weather(DEFAULT_CITY)
        avg_temp = get_average_temperature_for_location(3)

        return (
            f"{base_answer}\n\n"
            f"{weather}\n"
            f"{avg_temp}"
        )


    logger.warning("No match found for question: %s", question)
    return "Sorry, I don't recognize that question."


def semantic_suggestions(keyword, top_k=5):
    """Return top-k similar known questions based on a short keyword."""
    global QUESTION_LIST, QUESTION_EMBEDDINGS
    if QUESTION_EMBEDDINGS is None:
        rebuild_embeddings()

    # Embed keyword
    key_vec = next(EMBED_MODEL.embed([keyword]))

    # Compute cosine similarity with all questions
    scores = QUESTION_EMBEDDINGS @ key_vec / (
        np.linalg.norm(QUESTION_EMBEDDINGS, axis=1) * np.linalg.norm(key_vec)
    )

    # Top K results
    top_indices = np.argsort(scores)[::-1][:top_k]

    suggestions = []
    for idx in top_indices:
        if scores[idx] > 0.45:  # lower threshold for suggestions
            suggestions.append(QUESTION_LIST[idx])

    return suggestions

def split_compound_question(text):
    """Split multiple questions based on connectors."""
    t = text.lower().strip()

    connectors = [" and ", " & ", " also ", " plus ", ", then "]
    for c in connectors:
        t = t.replace(c, "? ")

    parts = [p.strip() for p in t.split("?") if p.strip()]
    return parts


def chat_mode():
    show_symbol(START_SYMBOL)
    print(f"{current_time()} Hello!")
    time.sleep(1)
    print(f"{current_time()} How can I help you? (Type 'bye' to exit)")
    suggestions = None
    show_temperature_idle()

    while True:
        log_raw_temperature()
        user_input = input(f"{current_time()} ").strip()

        if user_input.lower() == "bye":
            print(f"{current_time()} Goodbye!")
            break

        if user_input.lower() == "help":
            print(f"{current_time()} You can ask about:\n" + "\n".join(known_questions().keys()))
            continue

        if user_input.lower() == "trivia":
            show_symbol(GAME_START_SYMBOL)
            trivia_game()
            print(f"{current_time()} Trivia finished. Resuming normal chat...\n")
            continue    
        
        # numeric selection from suggestion list
        if suggestions is not None and user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(suggestions):
                selected = suggestions[idx]
                print(f"{current_time()} {get_answer(selected)}")
            else:
                print(f"{current_time()} Invalid selection.")
            suggestions = None
            continue

        # compound question detection
        parts = split_compound_question(user_input)
        if len(parts) > 1:
            for p in parts:
                print(f"{current_time()} Q: {p}")
                print(f"{current_time()} A: {get_answer(p)}")
            continue

        # keyword suggestion mode
        words = user_input.split()
        if "?" not in user_input and len(words) <= 3:
            suggestions_list = semantic_suggestions(user_input)
            if suggestions_list:
                suggestions = suggestions_list
                print(f"{current_time()} I found related questions:")
                for i, q in enumerate(suggestions, 1):
                    print(f"{current_time()} {i}. {q}")
                print(f"{current_time()} Choose a number.")
                continue

        # normal semantic match
        suggestions = None
        print(f"{current_time()} {get_answer(user_input)}")


def direct_mode(question):
    print(f"{current_time()} {get_answer(question)}")


class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # Called when user passes unknown or invalid arguments
        print(f"\nERROR: {message}\n")
        self.print_help()   # show help text
        exit(2)             # exit instead of running chat mode

def trivia_game():
    print(f"{current_time()} Trivia mode activated! Type 'trivia' again to exit the game early.\n")

    score = 0
    asked = 0
    total_questions = 10

    # Make a shuffled copy so we don't modify original
    questions = random.sample(TRIVIA_QUESTIONS, len(TRIVIA_QUESTIONS))

    while asked < total_questions:

        print(f"{current_time()} " 
              f"Question {asked + 1} of {total_questions}; "
              f"Score {score}/{total_questions}")

        q = questions[asked % len(questions)]
        print(f"{current_time()} Q{asked+1}: {q['question']}")
        for opt in q["options"]:
            print(f"   {opt}")

        user_answer = input(f"{current_time()} Your answer (A/B/C/D): ").strip().upper()

        # Exit trivia mode
        if user_answer.lower() == "trivia":

            show_score_on_led(f'{score}/{asked}')
            show_symbol(GAME_EXIT_SYMBOL)   
            print(f"{current_time()} Exiting Trivia mode early.")
            print(f"{current_time()} Your score: {score}/{asked}\n")
            return

        # Validate
        # if user_answer == q["correct"]:
        #     print(f"{current_time()} Correct!\n")
        #     score += 1
        # else:
        #     print(f"{current_time()} Incorrect! Correct answer: {q['correct']}\n")

        if user_answer == q["correct"]:
            print(f"{current_time()} Correct!\n")
            show_correct_symbol()
            score += 1
        else:
            print(f"{current_time()} Incorrect! Correct answer: {q['correct']}\n")
            show_wrong_symbol()

        asked += 1


    print(f"{current_time()} Trivia complete! Your final score: {score}/{total_questions}\n")

def setup_logging(enable_log: bool, log_level: str, log_file: str, enable_debug: bool):
    """
    Unified logging setup:
    - Console shows DEBUG if --debug is used, WARNING otherwise.
    - Log file stores only INFO and WARNING messages when --log is enabled.
    - Debug messages are NOT written to log file (not required by assignment).
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Master level, handlers decide what to output

    # Remove old handlers to prevent duplicates
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    # ---------------------------
    # Console Handler
    # ---------------------------
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    if enable_debug:
        console_handler.setLevel(logging.DEBUG)   # Show detailed debug info on console
    else:
        console_handler.setLevel(logging.WARNING) # Normal run → only warnings on console

    root.addHandler(console_handler)

    # ---------------------------
    # File Handler (only if --log is set)
    # ---------------------------
    if enable_log:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        file_handler.setLevel(getattr(logging, log_level.upper()))  # INFO or WARNING
        root.addHandler(file_handler)

    return root


def main():
    # parser = argparse.ArgumentParser(description="TerminalTalk - A Terminal Chatbot")
    parser = CustomArgumentParser(description="TerminalTalk - A Terminal Chatbot")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--add",action="store_true",help="Add a question/answer to the internal list",)
    mode_group.add_argument("--remove",action="store_true",help="Remove a question from the internal list",)

    parser.add_argument("--question",help="Ask a question in direct mode, or specify the question for --add/--remove",)
    parser.add_argument("--answer",action="append",help="Answer text for --add. Use multiple --answer options for multiple answers.",)
    
    parser.add_argument("--import",dest="import_mode",action="store_true",help="Import questions/answers from a file instead of using internal ones",)
    parser.add_argument("--filetype",type=str,default="CSV",help="Type of import file (currently only CSV is supported)",)
    parser.add_argument("--filepath",type=str,help="Path to the import file (e.g. ./qa.csv)",)
    
    parser.add_argument("--list-questions",action="store_true",help="List all known questions and exit",)
    
    parser.add_argument("--log", action="store_true", help="Enable logging")
    parser.add_argument("--log-level", default="WARNING", choices=["INFO", "WARNING"], help="Logging level")
    parser.add_argument("--log-file", default="terminaltalk.log", help="Log file path")

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for developers (prints internal diagnostic information)"
    )
    parser.add_argument(
        "--temp-diff",
        action="store_true",
        help="Show temperature increase/decrease (max-min) for last 3 days (local + forecast)"
    )

    args = parser.parse_args()
    if args.temp_diff:
        log_local_temperature()
        result = get_last_3_days_temperature_change()
        print(f"{current_time()}  {result}")
        return


    # Setup logging once, correctly
    logger = setup_logging(
        enable_log=args.log,
        log_level=args.log_level,
        log_file=args.log_file,
        enable_debug=args.debug
    )

    if args.debug:
        print(f"{current_time()} DEBUG MODE ENABLED")
        logger.debug("Debugging mode initialized.")

    if args.log:
        logger.info("Logging enabled at level %s (file: %s)", args.log_level, args.log_file)

    if args.import_mode:
        if not args.filepath:
            print(f"{current_time()} ERROR: --import was used but no --filepath was provided.")
            print(f"{current_time()} Using internal questions instead.")
        else:
            safe_import_csv(args.filepath)
    
    if args.add:
        if not args.question:
            print(f"{current_time()} ERROR: --add requires --question.")
            return
        if not args.answer:
            print(f"{current_time()} ERROR: --add requires at least one --answer.")
            return
        add_question_cli(args.question, args.answer)
        return


    if args.remove:
        if not args.question:
            print(f"{current_time()} ERROR: --remove requires --question.")
            return
        remove_question_cli(args.question, args.answer)
        return


    if args.list_questions:
        return list_questions()
    
    # Build embeddings before answering
    rebuild_embeddings()
    if args.question:
        direct_mode(args.question)
    else:
        chat_mode()


if __name__ == "__main__":
    main()


