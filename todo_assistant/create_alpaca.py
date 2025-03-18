import json
import re
from typing import Optional, List, Dict
from datetime import datetime

def remove_links(text: str) -> str:
    """Remove URLs from text using regex."""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(url_pattern, '', text).strip()

def swap_datetime_if_needed(task: Dict) -> Dict:
    """Swap dueDateTime and reminderDateTime if reminder is after due date."""
    if ("dueDateTime" in task and task["dueDateTime"].get("dateTime") and
        "reminderDateTime" in task and task["reminderDateTime"].get("dateTime")):
        due_dt = datetime.strptime(task["dueDateTime"]["dateTime"][:19], "%Y-%m-%dT%H:%M:%S")
        reminder_dt = datetime.strptime(task["reminderDateTime"]["dateTime"][:19], "%Y-%m-%dT%H:%M:%S")
        if due_dt < reminder_dt:
            task["dueDateTime"], task["reminderDateTime"] = task["reminderDateTime"], task["dueDateTime"]
    return task

def remove_duplicates(tasks: List[Dict]) -> List[Dict]:
    """Remove duplicate tasks based on title and checklist items."""
    seen = set()
    unique_tasks = []
    
    for task in tasks:
        checklist = tuple(item["displayName"] for item in task.get("checklistItems", []))
        identifier = (task["title"], checklist)
        if identifier not in seen:
            seen.add(identifier)
            unique_tasks.append(task)
    return unique_tasks

def filter_tasks(tasks: List[Dict]) -> List[Dict]:
    """Filter tasks based on specified criteria."""
    filtered_tasks = []
    for task in tasks:
        has_checklist = "checklistItems" in task and len(task["checklistItems"]) > 0
        has_due_date = "dueDateTime" in task and task["dueDateTime"].get("dateTime")
        has_reminder = "reminderDateTime" in task and task["reminderDateTime"].get("dateTime")
        
        # Skip tasks with no subtasks and no dates
        if not has_checklist and not has_due_date and not has_reminder:
            continue
            
        # Skip tasks with no subtasks if they exist
        if not has_checklist:
            continue
            
        filtered_tasks.append(task)
    return filtered_tasks

def process_task(task: Dict) -> Optional[Dict]:
    """Process individual task: remove links and swap dates if needed."""
    try:
        # Remove links from title
        task["title"] = remove_links(task["title"])
        
        # Remove links from checklist items
        if "checklistItems" in task:
            for item in task["checklistItems"]:
                item["displayName"] = remove_links(item["displayName"])
        
        # Swap dates if needed
        task = swap_datetime_if_needed(task)
        
        return task
    except Exception as e:
        print(f"Warning: Error processing task '{task.get('title', 'Unknown')}': {e}")
        return None

def create_alpaca_dataset(
    instruction_path: str = 'config/todo.txt',
    tasks_path: str = 'config/all_tasks.json',
    output_path: str = 'test.json',
    max_tasks: Optional[int] = None
) -> None:
    """Create an Alpaca dataset from tasks and save it as a JSON file."""
    # Read instruction
    try:
        with open(instruction_path, 'r', encoding='utf-8') as f:
            instruction = f.read().strip()
    except FileNotFoundError:
        print(f"Error: Instruction file '{instruction_path}' not found.")
        return

    # Load tasks
    try:
        with open(tasks_path, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
    except FileNotFoundError:
        print(f"Error: Tasks file '{tasks_path}' not found.")
        return

    # Process tasks
    tasks = remove_duplicates(tasks)
    tasks = filter_tasks(tasks)
    tasks = [process_task(task) for task in tasks if process_task(task) is not None]
    tasks = tasks[:max_tasks] if max_tasks else tasks

    # Create dataset
    alpaca_dataset = []
    for task in tasks:
        try:
            created_time = datetime.strptime(task["createdDateTime"][:19], "%Y-%m-%dT%H:%M:%S")
            formatted_time = created_time.strftime("%d-%m-%Y %H:%M:%S")
            input_str = f"Task: {task['title']} which is created at {formatted_time}"

            output_task = {
                "importance": task.get("importance", "normal"),
                "title": task["title"],
                "reminderDateTime": task.get("reminderDateTime", {"dateTime": ""}),
                "dueDateTime": task.get("dueDateTime", {"dateTime": ""}),
                "checklistItems": [{"displayName": item["displayName"]} for item in task.get("checklistItems", [])]
            }

            output_str = f"```json\n{json.dumps(output_task, ensure_ascii=False, indent=4)}\n```"
            alpaca_dataset.append({"instruction": instruction, "input": input_str, "output": output_str})
        except Exception as e:
            print(f"Warning: Error processing task '{task.get('title', 'Unknown')}': {e}")

    # Write output
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(alpaca_dataset, f, ensure_ascii=False, indent=2)
        print(f"Successfully created dataset at '{output_path}' with {len(alpaca_dataset)} entries.")
    except Exception as e:
        print(f"Error writing to output file '{output_path}': {e}")

if __name__ == "__main__":
    instruction_path = "config/todo.txt"
    tasks_path = "data/all_tasks.json"
    output_path = "todo_assistant/todo_data.json"

    create_alpaca_dataset(
        instruction_path=instruction_path,
        tasks_path=tasks_path,
        output_path=output_path,
    )
