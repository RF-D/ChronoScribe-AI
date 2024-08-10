import os
import time
import schedule
from datetime import datetime
from termcolor import colored
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
import concurrent.futures
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import hashlib
import json

# Initialize Ollama
llm = Ollama(model="llama3.1")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# File to store metadata about processed files
metadata_file = "file_metadata.json"

# Define the default prompt template
default_prompt_template = """
You are an AI assistant tasked with analyzing files in a specified folder and generating a report.
Please analyze the contents of the following files and create a comprehensive report:

{file_contents}

Your report should include:
1. A summary of each file's content
2. Key insights or patterns observed across the files
3. Any anomalies or interesting findings
4. Recommendations based on the analysis

Please format your report in markdown.
"""


def print_colored(text, color="white", on_color=None, attrs=None):
    print(colored(text, color, on_color, attrs))


def get_user_prompt():
    print_colored("\nEnter the role and task for the AI assistant.", color="cyan")
    print_colored("Press Enter without typing to use the default prompt.", color="cyan")

    role = input(
        colored(
            "Enter the role for the AI assistant (e.g., 'data analyst', 'code reviewer'): ",
            "yellow",
        )
    ).strip()
    task = input(colored("Enter the task for the AI assistant: ", "yellow")).strip()

    if not role and not task:
        print_colored("Using default prompt.", color="green")
        return default_prompt_template

    custom_prompt_template = f"""
You are an AI assistant acting as a {role or 'general assistant'}.
Your task is to {task or 'analyze the following file contents'}:

{{file_contents}}

Please provide your analysis or complete the task as specified.
Format your response in markdown.
"""
    print_colored("Custom prompt created.", color="green")
    return custom_prompt_template


def calculate_file_hash(file_path):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_folder_hash(folder_path):
    """Generate a unique hash for the folder path."""
    return hashlib.md5(folder_path.encode()).hexdigest()[:10]


def load_file_metadata(metadata_file):
    """Load metadata of previously processed files."""
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            return json.load(f)
    return {}


def save_file_metadata(metadata, metadata_file):
    """Save metadata of processed files."""
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)


def process_and_store_documents(folder_path):
    file_contents = read_file_contents(folder_path)
    if not file_contents:
        print_colored(
            "No readable files found in the specified folder.", color="yellow"
        )
        return None

    folder_hash = get_folder_hash(folder_path)
    persist_directory = f"./chroma_db_{folder_hash}"

    # Initialize Chroma with the folder-specific persist directory
    vectorstore = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.create_documents([file_contents])

    # Load existing metadata for this folder
    metadata_file = f"file_metadata_{folder_hash}.json"
    metadata = load_file_metadata(metadata_file)
    current_version = metadata.get("version", 0)
    files_metadata = metadata.get("files", {})

    # Check for changes and update only if necessary
    changes_made = False
    for doc in documents:
        file_path = doc.metadata.get("source")
        if file_path:
            current_hash = calculate_file_hash(file_path)
            if files_metadata.get(file_path) != current_hash:
                files_metadata[file_path] = current_hash
                changes_made = True

    if changes_made or current_version == 0:
        current_version += 1
        metadata["version"] = current_version
        metadata["files"] = files_metadata
        metadata["last_updated"] = datetime.now().isoformat()

        # Update vector store
        vectorstore.add_documents(documents)
        print_colored(f"Updated vector store for folder {folder_path}", color="green")
        print_colored(
            f"Added {len(documents)} documents to the vector store.", color="green"
        )

        # Save updated metadata
        save_file_metadata(metadata, metadata_file)
    else:
        print_colored(
            f"No changes detected for folder {folder_path}. Using existing vector store.",
            color="cyan",
        )

    return persist_directory, current_version


def generate_report(folder_path, custom_prompt):
    result = process_and_store_documents(folder_path)
    if result is None:
        return

    persist_directory, current_version = result

    prompt = PromptTemplate(input_variables=["file_contents"], template=custom_prompt)
    chain = prompt | llm | (lambda x: x.content if isinstance(x, HumanMessage) else x)

    print_colored(f"Generating report using version {current_version}...", color="cyan")

    # Use the folder-specific vector store
    vectorstore = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )

    # Retrieve relevant documents from the vector store
    query = "Summarize the key points from all documents"
    relevant_docs = vectorstore.similarity_search(query, k=5)

    if not relevant_docs:
        print_colored(
            "No relevant documents found in the vector store.", color="yellow"
        )
        return

    full_report = []
    for doc in relevant_docs:
        try:
            result = chain.invoke({"file_contents": doc.page_content})
            full_report.append(result)
            print_colored("Processed document chunk", color="green")
        except Exception as e:
            print_colored(f"Document processing failed: {e}", color="red")

    if not full_report:
        print_colored("No content was generated for the report.", color="yellow")
        return

    full_report_text = "\n\n".join(full_report)

    folder_hash = get_folder_hash(folder_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"report_{folder_hash}_v{current_version}_{timestamp}.md"
    with open(report_filename, "w") as report_file:
        report_file.write(full_report_text)

    print_colored(
        f"Full report for folder {folder_path} (version {current_version}) generated and saved as {report_filename}",
        color="green",
    )
    print_colored(
        f"Report content preview:\n{full_report_text[:500]}...", color="cyan"
    )  # Print first 500 characters of the report


def sanitize_path(path):
    # Expand user directory if path starts with ~
    path = os.path.expanduser(path)
    # Strip leading and trailing whitespace
    path = path.strip()
    # Resolve any symbolic links and normalize the path
    path = os.path.realpath(path)
    return path


def read_file_contents(folder_path):
    folder_path = sanitize_path(folder_path)

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"'{folder_path}' is not a directory.")

    file_contents = ""
    files_found = False
    print_colored(f"Scanning folder: {folder_path}", color="cyan")

    supported_extensions = (
        ".py",
        ".js",
        ".ts",
        ".html",
        ".css",
        ".scss",
        ".json",
        ".yaml",
        ".yml",
        ".md",
        ".txt",
        ".csv",
        ".xml",
        ".sql",
        ".sh",
        ".bash",
        ".env",
        ".gitignore",
        ".dockerignore",
        "Dockerfile",
        "docker-compose.yml",
        ".jsx",
        ".tsx",
        ".vue",
        ".php",
        ".rb",
        ".go",
        ".java",
        ".kt",
        ".swift",
    )

    def process_file(file_path, relative_path):
        nonlocal file_contents, files_found
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                file_contents += f"File: {relative_path}\n\n{content}\n\n---\n\n"
                files_found = True
                print_colored(f"Read file: {relative_path}", color="green")
        except Exception as e:
            print_colored(f"Error reading file {relative_path}: {str(e)}", color="red")

    def scan_directory(current_path, relative_path=""):
        for item in os.listdir(current_path):
            item_path = os.path.join(current_path, item)
            item_relative_path = os.path.join(relative_path, item)

            if os.path.isfile(item_path):
                if item.lower().endswith(supported_extensions) or "." not in item:
                    process_file(item_path, item_relative_path)
                else:
                    print_colored(
                        f"Skipped file: {item_relative_path} (unsupported extension)",
                        color="yellow",
                    )
            elif os.path.isdir(item_path):
                print_colored(f"Entering directory: {item_relative_path}", color="cyan")
                scan_directory(item_path, item_relative_path)

    scan_directory(folder_path)

    if not files_found:
        print_colored("No readable files found with supported extensions.", color="red")

    return file_contents


def schedule_task(folder_path, interval_minutes, custom_prompt):
    def task():
        generate_report(folder_path, custom_prompt)

    schedule.every(interval_minutes).minutes.do(task)
    print_colored(
        f"Task scheduled to run every {interval_minutes} minutes. Press Ctrl+C to stop.",
        color="green",
    )

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    print_colored(
        "Welcome to the AI Report Generator!", color="magenta", attrs=["bold"]
    )

    folder_path = input(colored("Enter the folder path to monitor: ", "yellow"))
    folder_path = os.path.expanduser(folder_path)

    custom_prompt = get_user_prompt()

    execution_choice = input(
        colored(
            "Do you want to (1) run the task instantly or (2) schedule it? Enter 1 or 2: ",
            "yellow",
        )
    )

    try:
        if execution_choice == "1":
            print_colored("Running the task instantly...", color="green")
            generate_report(folder_path, custom_prompt)
        elif execution_choice == "2":
            interval_minutes = int(
                input(
                    colored(
                        "Enter the interval in minutes for recurring execution: ",
                        "yellow",
                    )
                )
            )
            print_colored(
                f"Scheduling task to run every {interval_minutes} minutes.",
                color="green",
            )
            print_colored(
                "Running the task instantly before scheduling...", color="cyan"
            )
            generate_report(folder_path, custom_prompt)  # Run instantly
            schedule_task(folder_path, interval_minutes, custom_prompt)
        else:
            print_colored("Invalid choice. Exiting.", color="red")
    except Exception as e:
        print_colored(f"An error occurred: {str(e)}", color="red")
