import time
import argparse

def current_time():
    return time.strftime("%H:%M:%S")

def known_questions():
    return {
        "what is your name?": "I'm TerminalTalk, your terminal assistant!",
        "what is a variable?": "A variable is a named location in memory used to store data for your program.",
        "what is a function?": "A function is a block of reusable code that performs a specific task.",
        "what is oop?": "OOP stands for Object-Oriented Programming — a way to structure code using classes and objects.",
        "what is git?": "Git is a version control system used to track changes in your code.",
        "what is api?": "API stands for Application Programming Interface — a way for programs to communicate with each other.",
        "what is debugging?": "Debugging is the process of finding and fixing errors in your code.",
        "what is machine learning?": "Machine learning is a type of AI that allows systems to learn from data and improve over time.",
        "what is json?": "JSON stands for JavaScript Object Notation — a lightweight format for storing and transferring data.",
        "what is a loop?": "A loop is a control structure that repeats a block of code multiple times.",
        "what is recursion?": "Recursion is when a function calls itself to solve smaller instances of a problem."
    }

def get_answer(question):
    question = question.lower().strip()
    qa_dict = known_questions()

    for key, answer in qa_dict.items():
        if key in question:
            return answer
    return "Sorry, I don't recognize that question. Please ask another one."

def chat_mode():
    print(f"{current_time()} Hello!")
    time.sleep(1)
    print(f"{current_time()} How can I help you? (Type 'bye' to exit)")

    while True:
        user_input = input(f"{current_time()} ").strip().lower()

        if user_input == "bye":
            print(f"{current_time()} Goodbye!")
            break
        elif user_input == "help":
            print(f"{current_time()} You can ask about:\n" + "\n".join(known_questions().keys()))
        else:
            print(f"{current_time()} {get_answer(user_input)}")

def direct_mode(question):
    print(f"{current_time()} {get_answer(question)}")

def main():
    parser = argparse.ArgumentParser(description="TerminalTalk - A Terminal Chatbot")
    parser.add_argument("--question")
    args = parser.parse_args()

    if args.question:
        direct_mode(args.question)
    else:
        chat_mode()

if __name__ == "__main__":
    main()
