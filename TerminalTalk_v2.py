import difflib
import time
import argparse
import random
import csv
import os
import json 

IMPORTED_QA = None

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
    "which floor is lecture hall 203 on?": [
        "Lecture hall 203 is on the second floor. Please check the room signs next to the staircase or elevator.",
        "Lecture hall 203 is on an upper floor. You can confirm the exact floor from the building map near the entrance."
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
        "You can collect your student ID card from the admissions office."
    ]
}

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


""" Return the currently active questions:
    - If a CSV was imported, use IMPORTED_QA.
    - Otherwise, use INTERNAL_QA.
"""
def known_questions():
    global IMPORTED_QA, QA_DATA

    if IMPORTED_QA is not None:
        return IMPORTED_QA
    
    return QA_DATA


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


def get_answer(question):
    question_norm = question.lower().strip()
    qa_dict = known_questions()
    matched_answers = []

    # try your existing logic (exact / substring match)
    for key, answers in qa_dict.items():
        if key in question_norm:
            if isinstance(answers, list):
                matched_answers.append(random.choice(answers))
            else:
                matched_answers.append(answers)

    # If we found matches with the old method, just return them
    if matched_answers:
        return "\n".join(matched_answers)

    # use fuzzy matching to handle variants (similar questions)
    # get_close_matches returns a list of closest strings
    # cutoff = 0.6 means “at least 60% similar”
    all_keys = list(qa_dict.keys())
    close = difflib.get_close_matches(question_norm, all_keys, n=1, cutoff=0.6)

    if close:
        canonical_key = close[0]
        answers = qa_dict[canonical_key]

        if isinstance(answers, list):
            return random.choice(answers)
        return answers

    return "Sorry, I don't recognize that question. Please ask another one."


def chat_mode():
    print(f"{current_time()} Hello!")
    time.sleep(1)
    print(f"{current_time()} How can I help you? (Type 'bye' to exit)")
    suggestions = None

    while True:
        user_input = input(f"{current_time()} ").strip().lower()
        words = user_input.split()
        
        if user_input == "bye":
            print(f"{current_time()} Goodbye!")
            break

        if user_input == "help":
            print(f"{current_time()} You can ask about:\n" + "\n".join(known_questions().keys()))
            continue

        if suggestions is not None and user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(suggestions):
                selected_question = suggestions[idx]
                print(f"{current_time()} You selected: {selected_question}")
                print(f"{current_time()} {get_answer(selected_question)}")
            else:
                print(f"{current_time()} Invalid selection. Please choose a valid number from the list.")
            suggestions = None
            continue

        if "?" not in user_input and len(words) <= 2:
            keyword = user_input
            all_questions = list(known_questions().keys())
            matches = [q for q in all_questions if keyword in q.lower()]

            if matches:
                suggestions = matches
                print(f"{current_time()} Here are some questions related to '{keyword}':")
                for i, q in enumerate(suggestions, start=1):
                    print(f"{current_time()} {i}. {q}")
                print(f"{current_time()} Please select a question by typing its number.")
                continue

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
    
    if args.question:
        direct_mode(args.question)
    else:
        chat_mode()


if __name__ == "__main__":
    main()


