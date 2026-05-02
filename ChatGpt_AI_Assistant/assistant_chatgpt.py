import os
import tkinter as tk
from tkinter import scrolledtext
import threading
import speech_recognition as sr
import pyttsx3
from openai import OpenAI

# ----------------- OpenAI Client -----------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------- Text To Speech -----------------
engine = pyttsx3.init()
engine.setProperty("rate", 170)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ----------------- Speech Recognition -----------------
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        status_label.config(text="🎤 Listening...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        status_label.config(text="✅ Recognized")
        return text
    except:
        status_label.config(text="❌ Could not understand")
        return ""

# ----------------- ChatGPT Logic -----------------
conversation = [
    {"role": "system", "content": "You are a helpful AI desktop assistant."}
]

def chat_with_gpt(user_text):
    conversation.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation,
        temperature=0.7,
        max_completion_tokens=200
    )

    reply = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": reply})
    return reply

# ----------------- Assistant Run -----------------
def run_assistant():
    user_text = listen()
    if not user_text:
        return

    chat_box.insert(tk.END, f"You: {user_text}\n")

    reply = chat_with_gpt(user_text)

    chat_box.insert(tk.END, f"Assistant: {reply}\n\n")
    chat_box.see(tk.END)
    speak(reply)

# ----------------- Thread Wrapper -----------------
def start():
    threading.Thread(target=run_assistant).start()

# ----------------- Tkinter GUI -----------------
root = tk.Tk()
root.title("ChatGPT AI Desktop Assistant")
root.geometry("540x520")
root.resizable(False, False)

title = tk.Label(root, text="🤖 ChatGPT Desktop Assistant",
                 font=("Arial", 16, "bold"))
title.pack(pady=10)

chat_box = scrolledtext.ScrolledText(root, width=60, height=20)
chat_box.pack(pady=10)

status_label = tk.Label(root, text="Click Listen to talk",
                        fg="blue")
status_label.pack()

listen_btn = tk.Button(root, text="🎧 Listen",
                       font=("Arial", 12),
                       command=start)
listen_btn.pack(pady=10)

exit_btn = tk.Button(root, text="Exit",
                     command=root.destroy)
exit_btn.pack()

speak("Hello! I am your ChatGPT powered desktop assistant.")
root.mainloop()