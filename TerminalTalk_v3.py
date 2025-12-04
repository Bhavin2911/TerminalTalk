# import difflib
import time
import argparse
import random
import csv
import os
import json 

# -------- FASTEMBED semantic embeddings --------
from fastembed import TextEmbedding
import numpy as np

# Initialize fastembed model
EMBED_MODEL = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")


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


def load_questions_from_csv(filepath: str):

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
    global QUESTION_LIST, QUESTION_EMBEDDINGS

    q = question.lower().strip()
    qa_dict = known_questions()

    # 1) Exact or substring match (old behavior preserved)
    for key, answers in qa_dict.items():
        if key == q or key in q or q in key:
            return random.choice(answers)

    # 2) Semantic fastembed match
    if QUESTION_EMBEDDINGS is None:
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

    if best_score >= SIMILARITY_THRESHOLD:
        return random.choice(qa_dict[best_question])

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
    print(f"{current_time()} Hello!")
    time.sleep(1)
    print(f"{current_time()} How can I help you? (Type 'bye' to exit)")
    suggestions = None

    while True:
        user_input = input(f"{current_time()} ").strip()

        if user_input.lower() == "bye":
            print(f"{current_time()} Goodbye!")
            break

        if user_input.lower() == "help":
            print(f"{current_time()} You can ask about:\n" + "\n".join(known_questions().keys()))
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


def main():
    parser = argparse.ArgumentParser(description="TerminalTalk - A Terminal Chatbot")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--add",action="store_true",help="Add a question/answer to the internal list",)
    mode_group.add_argument("--remove",action="store_true",help="Remove a question from the internal list",)

    parser.add_argument("--question",help="Ask a question in direct mode, or specify the question for --add/--remove",)
    parser.add_argument("--answer",action="append",help="Answer text for --add. Use multiple --answer options for multiple answers.",)
    
    parser.add_argument("--import",dest="import_mode",action="store_true",help="Import questions/answers from a file instead of using internal ones",)
    parser.add_argument("--filetype",type=str,default="CSV",help="Type of import file (currently only CSV is supported)",)
    parser.add_argument("--filepath",type=str,help="Path to the import file (e.g. ./qa.csv)",)
    
    parser.add_argument("--list-questions",action="store_true",help="List all known questions and exit",)
    args = parser.parse_args()

    if args.import_mode:
        if not args.filepath:
            print(f"{current_time()} ERROR: --import was used but no --filepath was provided.")
            print(f"{current_time()} Using internal questions instead.")
        else:
            if args.filetype.lower() == "csv":
                load_questions_from_csv(args.filepath)
            else:
                print(f"{current_time()} ERROR: Unsupported filetype '{args.filetype}'. Only 'CSV' is supported.")
                print(f"{current_time()} Using internal questions instead.")

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


