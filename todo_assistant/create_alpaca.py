import sys
sys.path.append("")

import json
import re
from typing import Optional, List, Dict
from datetime import datetime, timezone, timedelta
import requests
from collections import defaultdict
import os

# Retrieve OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")

# Define the OpenAI API endpoint
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Define the path to the prompt file
PROMPT_PATH = "todo_assistant/config/generate_date_prompt.txt"

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
    """Filter tasks with checklist items."""
    return [task for task in tasks if "checklistItems" in task and len(task["checklistItems"]) > 0]

def extract_json(text: str) -> list:
    """Extract JSON content from text with debugging."""
    text = re.sub(r'```json\n|\n```', '', text).strip()
    start_index = text.find('[')
    end_index = text.rfind(']') + 1
    json_string = text[start_index:end_index]
    try:
        result = json.loads(json_string)
        return result
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Generated text: {text}")
        return []

def process_task(task: Dict) -> Optional[Dict]:
    """Process individual task: remove links and swap dates if needed."""
    try:
        task["title"] = remove_links(task["title"])
        if "checklistItems" in task:
            for item in task["checklistItems"]:
                item["displayName"] = remove_links(item["displayName"])
        task = swap_datetime_if_needed(task)
        return task
    except Exception as e:
        print(f"Warning: Error processing task '{task.get('title', 'Unknown')}': {e}")
        return None

def save_dataset(dataset: List[Dict], output_path: str):
    """Save the dataset to a JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved dataset with {len(dataset)} entries to '{output_path}'.")
    except Exception as e:
        print(f"Error writing to output file '{output_path}': {e}")

def read_prompt_from_file(path: str) -> str:
    """Read the prompt from a file."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read().strip()

def convert_to_hanoi_time(utc_dt: datetime) -> datetime:
    """Convert UTC datetime to Hanoi time (UTC+7)."""
    hanoi_offset = timedelta(hours=7)
    return utc_dt.replace(tzinfo=timezone.utc).astimezone(timezone(hanoi_offset))

def get_AI_dates(tasks: List[Dict]) -> List[Dict]:
    """Generate dates using OpenAI API, considering importance."""
    if not tasks:
        return tasks
    
    try:
        prompt = read_prompt_from_file(PROMPT_PATH)
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return tasks
    
    task_details = ""
    for i, task in enumerate(tasks):
        created_dt = datetime.strptime(task["createdDateTime"][:19], "%Y-%m-%dT%H:%M:%S")
        hanoi_dt = convert_to_hanoi_time(created_dt)
        importance = task.get("importance", "normal")
        task_details += f"{i+1}. Title: {task['title']}, Importance: {importance}, Created: {hanoi_dt.strftime('%Y-%m-%dT%H:%M:%SZ')} ({hanoi_dt.strftime('%A')})\n"
    
    prompt += "\nHere are the tasks:\n" + task_details
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        generated_text = response_data["choices"][0]["message"]["content"]
        generated_dates = extract_json(generated_text)
        
        if len(generated_dates) != len(tasks):
            print(f"Warning: Expected {len(tasks)} date assignments, got {len(generated_dates)}")
        
        for i, date_info in enumerate(generated_dates[:len(tasks)]):
            reminder_dt = datetime.strptime(date_info["reminderDateTime"][:19], "%Y-%m-%dT%H:%M:%S")
            due_dt = datetime.strptime(date_info["dueDateTime"][:19], "%Y-%m-%dT%H:%M:%S")
            
            tasks[i]["reminderDateTime"] = {"dateTime": reminder_dt.strftime('%Y-%m-%dT%H:%M:%S'), "timeZone": "Asia/Ho_Chi_Minh"}
            tasks[i]["dueDateTime"] = {"dateTime": due_dt.strftime('%Y-%m-%dT%H:%M:%S'), "timeZone": "Asia/Ho_Chi_Minh"}
            tasks[i] = swap_datetime_if_needed(tasks[i])
    except requests.RequestException as e:
        print(f"Error calling OpenAI API: {e}")
    except KeyError as e:
        print(f"Error parsing OpenAI response: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return tasks

def create_dataset_entry(task: Dict, instruction: str) -> Dict:
    """Create a dataset entry for a task."""
    created_time = datetime.strptime(task["createdDateTime"][:19], "%Y-%m-%dT%H:%M:%S")
    hanoi_time = convert_to_hanoi_time(created_time)
    formatted_time = hanoi_time.strftime("%d-%m-%Y %H:%M:%S")
    input_str = f"Task: {task['title']} which is created at {formatted_time} ({hanoi_time.strftime('%A')})"
    
    output_task = {
        "importance": task.get("importance", "normal"),
        "title": task["title"],
        "checklistItems": [{"displayName": item["displayName"]} for item in task.get("checklistItems", [])]
    }
    
    if task.get("reminderDateTime", {}).get("dateTime"):
        reminder_dt = datetime.strptime(task["reminderDateTime"]["dateTime"][:19], "%Y-%m-%dT%H:%M:%S")
        hanoi_reminder = convert_to_hanoi_time(reminder_dt)
        output_task["reminderDateTime"] = {"dateTime": hanoi_reminder.strftime("%Y-%m-%dT%H:%M:%S")}
    else:
        output_task["reminderDateTime"] = {"dateTime": ""}
        
    if task.get("dueDateTime", {}).get("dateTime"):
        due_dt = datetime.strptime(task["dueDateTime"]["dateTime"][:19], "%Y-%m-%dT%H:%M:%S")
        hanoi_due = convert_to_hanoi_time(due_dt)
        output_task["dueDateTime"] = {"dateTime": hanoi_due.strftime("%Y-%m-%dT%H:%M:%S")}
    else:
        output_task["dueDateTime"] = {"dateTime": ""}
    
    output_str = f"```json\n{json.dumps(output_task, ensure_ascii=False, indent=4)}\n```"
    return {"instruction": instruction, "input": input_str, "output": output_str}

def create_alpaca_dataset(
    instruction_path: str = 'config/todo.txt',
    tasks_path: str = 'config/all_tasks.json',
    output_path: str = 'test.json',
    max_tasks: Optional[int] = None
) -> None:
    """Create an Alpaca dataset from tasks and save it incrementally."""
    try:
        with open(instruction_path, 'r', encoding='utf-8') as f:
            instruction = f.read().strip()
    except FileNotFoundError:
        print(f"Error: Instruction file '{instruction_path}' not found.")
        return

    try:
        with open(tasks_path, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
    except FileNotFoundError:
        print(f"Error: Tasks file '{tasks_path}' not found.")
        return

    tasks = remove_duplicates(tasks)
    tasks = filter_tasks(tasks)
    tasks = [process_task(task) for task in tasks if process_task(task) is not None]
    tasks = tasks[:max_tasks] if max_tasks else tasks

    dataset = []

    for task in tasks:
        if task.get("reminderDateTime", {}).get("dateTime") and task.get("dueDateTime", {}).get("dateTime"):
            entry = create_dataset_entry(task, instruction)
            dataset.append(entry)

    tasks_needing_dates = [
        t for t in tasks 
        if not t.get("reminderDateTime", {}).get("dateTime") or not t.get("dueDateTime", {}).get("dateTime")
    ]

    tasks_by_day = defaultdict(list)
    for task in tasks_needing_dates:
        created_dt = datetime.strptime(task["createdDateTime"][:19], "%Y-%m-%dT%H:%M:%S")
        day = created_dt.date()
        tasks_by_day[day].append(task)

    priority_map = {"high": 2, "normal": 1, "low": 0}
    for day, day_tasks in tasks_by_day.items():
        day_tasks.sort(key=lambda x: (-priority_map[x.get("importance", "normal")], x["createdDateTime"]))
        
        for i in range(0, len(day_tasks), 5):
            batch = day_tasks[i:i+5]
            get_AI_dates(batch)  # Updates tasks in place
            for task in batch:
                entry = create_dataset_entry(task, instruction)
                dataset.append(entry)
            save_dataset(dataset, output_path)
    
    save_dataset(dataset, output_path)

if __name__ == "__main__":
    instruction_path = "config/todo.txt"
    tasks_path = "data/all_tasks.json"
    output_path = "todo_assistant/todo_data_2.json"
    create_alpaca_dataset(
        instruction_path=instruction_path,
        tasks_path=tasks_path,
        output_path=output_path,
        max_tasks=20,
    )
