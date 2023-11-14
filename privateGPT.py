import paramiko
import tkinter as tk
from constants import CHROMA_SETTINGS
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from tkinter import PhotoImage
import os
import argparse
import time


load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))


background_color = "#1565C0"  # Deep calm blue
text_color = "white"

load_dotenv()

def create_entry(window):
    return tk.Entry(window, font=("Helvetica", 12), bg=background_color, fg=text_color)

def connect_ssh():
    host = host_entry.get()
    username = username_entry.get()
    password = password_entry.get()

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=host, username=username, password=password)

    return ssh_client

def run_command():
    command = command_entry.get()
    ssh_client = connect_ssh()
    stdin, stdout, stderr = ssh_client.exec_command(command)
    output = stdout.read().decode()
    ssh_output.insert(tk.END, output)
    ssh_client.close()

root = tk.Tk()
root.title("Otter App")
root.configure(bg=background_color)

# Load Otter Image
otter_image = tk.PhotoImage(file="C://Users//david//OneDrive//Desktop//Stig App//STIG APP 4//privateGPT-main//privateGPT-main//source_documents//4SoifmQpAbHrGFzRqWAig-removebg-preview.png")
otter_label = tk.Label(root, image=otter_image, bg=background_color)
otter_label.pack(pady=10)

# SSH Connection
ssh_frame = tk.Frame(root, bg=background_color)
ssh_frame.pack(pady=10)

host_label = tk.Label(ssh_frame, text="Host:", bg=background_color, fg=text_color)
host_label.grid(row=0, column=0, padx=5)
host_entry = tk.Entry(ssh_frame)
host_entry.grid(row=0, column=1, padx=5)

username_label = tk.Label(ssh_frame, text="Username:", bg=background_color, fg=text_color)
username_label.grid(row=1, column=0, padx=5)
username_entry = tk.Entry(ssh_frame)
username_entry.grid(row=1, column=1, padx=5)

password_label = tk.Label(ssh_frame, text="Password:", bg=background_color, fg=text_color)
password_label.grid(row=2, column=0, padx=5)
password_entry = tk.Entry(ssh_frame, show="*")
password_entry.grid(row=2, column=1, padx=5)

command_label = tk.Label(ssh_frame, text="Command:", bg=background_color, fg=text_color)
command_label.grid(row=3, column=0, padx=5)
command_entry = tk.Entry(ssh_frame)
command_entry.grid(row=3, column=1, padx=5)

run_button = tk.Button(ssh_frame, text="Run Command", command=run_command, bg="#FFA726", fg=text_color)
run_button.grid(row=4, column=0, columnspan=2, pady=10)

ssh_output = tk.Text(root, wrap=tk.WORD, bg=background_color, fg=text_color)
ssh_output.pack(padx=10, pady=10)

def open_chat_window(qa):
    def send_query():
        query = user_input.get()
        start = time.time()
        res = qa(query)  # Assuming qa is a function that takes a query and returns a response
        answer = res['result']
        end = time.time()

        chat_history.insert(tk.END, f"User: {query}\n")
        chat_history.insert(tk.END, f"Model: {answer} (took {round(end - start, 2)} s.)\n")

    chat_window = tk.Toplevel(root)
    chat_window.title("Chat with NLP Model")
    chat_window.configure(bg=background_color)

    chat_history = tk.Text(chat_window, wrap=tk.WORD, bg=background_color, fg=text_color)
    chat_history.pack(padx=10, pady=10)

    user_input = tk.Entry(chat_window, bg=background_color, fg=text_color)
    user_input.pack()

    send_button = tk.Button(chat_window, text="Send", command=send_query, bg="#FFA726", fg=text_color)
    send_button.pack()

    chat_window.mainloop()



    

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = GPT4All(model='C:\\Users\\david\\OneDrive\\Desktop\\Stig App\\STIG APP 4\\privateGPT-main\\privateGPT-main\\Models\\ggml-gpt4all-j-v1.3-groovy.bin', backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model='C:\\Users\\david\\OneDrive\\Desktop\\Stig App\\STIG APP 4\\privateGPT-main\\privateGPT-main\\Models\\ggml-gpt4all-j-v1.3-groovy.bin', backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)

        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)

    open_chat_button = tk.Button(root, text="Open Chat", command=lambda: open_chat_window(qa))
    open_chat_button.pack()

    root.mainloop() 
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer = res['result']
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

        # Print the source documents
    def send_query():
        query = user_input.get()
        # Here you can call the function to get the response from the NLP model
        response = "Response from NLP model"  # Replace with actual response
        chat_history.insert(tk.END, f"User: {query}\n")
        chat_history.insert(tk.END, f"Model: {response}\n")

    send_button = tk.Button(chat_window, text="Send", command=send_query)
    send_button.pack()        

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

if __name__ == "__main__":
    main()
