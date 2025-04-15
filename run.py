# run.py
# Author: @ledondodo
# Project: LLMs for language learning - Difficulty Constrained Decoding
# EPFL dlab

from src.run_project import run_project

from algorithms.run_task_1 import run_task_1
from algorithms.run_task_2 import run_task_2
from algorithms.run_task_3 import run_task_3
from algorithms.run_task_4 import run_task_4
from algorithms.run_task_5 import run_task_5
from algorithms.run_task_6 import run_task_6
from algorithms.run_task_7 import run_task_7
from algorithms.run_task_8 import run_task_8
from algorithms.run_task_9 import run_task_9
from algorithms.run_task_10 import run_task_10

TASK_MAP = {
    1: run_task_1,
    2: run_task_2,
    3: run_task_3,
    4: run_task_4,
    5: run_task_5,
    6: run_task_6,
    7: run_task_7,
    8: run_task_8,
    9: run_task_9,
    10: run_task_10,
}


def run_tasks():
    """Prompt the user to select and run a task."""
    while True:
        task_input = input("\nChoose a task (1-10) or press Enter to exit: ").strip()
        if not task_input:
            print("Exiting task selection.")
            break
        if task_input.isdigit():
            task_number = int(task_input)
            if task_number in TASK_MAP:
                print(f"\nRunning Task {task_number}...")
                TASK_MAP[task_number]()  # Call the corresponding task function
            else:
                print("Invalid task number. Please choose between 1 and 10.")
        else:
            print(
                "Invalid input. Please enter a valid task number or press Enter to exit."
            )
        print("-" * 50)


def main():
    print("\n# LLMs for language learning: Difficulty Constrained Decoding")
    print("@ EPFL dlab")
    print("Author: Arthur Chansel")

    mode = input(
        "\nChoose an option:\n"
        "1. Run constrained decoding (final implementation)\n"
        "2. Run tasks (1-10)\n"
        "Press Enter to exit.\nYour choice: "
    ).strip()

    if mode == "1":
        run_project()
    elif mode == "2":
        run_tasks()
    elif mode == "":
        print("Exiting program.")
    else:
        print("Invalid input. Please choose 1, 2, or press Enter to exit.")


if __name__ == "__main__":
    main()
