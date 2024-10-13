from modules import adapter, speak
from prompts import prompts, dialogue
import environ
import random
import datetime


env = environ.Env()
environ.Env.read_env()

spk = speak.Speak(env)
ad = adapter.Adapter(env)
char = env("CHARACTER").lower()
char_prompt = getattr(prompts, char, "You are a helpful assistant.") + "\nAnswer the following request: {query}"
if env("SPEECH_ENABLED").lower() == "true":
    spk_enabled = True
else:
    spk_enabled = False

def log_data(data, file_path="./logs"):
    # Get the current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    file_name = f"{file_path}/log_{current_date}.txt"
    
    # Create a logs directory if it doesn't exist
    import os
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    # Open the file in append mode and write the data
    with open(file_name, 'a') as file:
        file.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {data}\n")
        
def speak_and_print(text):
    print(text)
    spk.stream(text) if spk_enabled else None
    
def go_to_sleep(text):
    if "back" in text.lower() and "sleep" in text.lower():
        go_sleep = f"{random.choice(dialogue.sleep)}"
        log_data(f"Dr. Brain | {go_sleep}")
        speak_and_print(go_sleep)
        return True
        
while True:
    #loop 1
    answer = False
    text = spk.transcribe()
    if text and env("WAKE_WORD").lower() in text.lower() and env("CHARACTER").lower() in text.lower():
        if "wake" in text.lower():
            print("Waking up...")
            log_data(f"User | {text}")
            wake_up = f"{random.choice(dialogue.wake)} ... Astra Mor Technician, who approaches? ... A new test subject?"
            log_data(f"Dr. Brain | {wake_up}")
            speak_and_print(wake_up)
            #!loop 2
            while answer is False:
                decision = spk.transcribe()
                if decision:
                    log_data(f"User | {decision}")
                    if "yes" in decision.lower():
                        subject_number = random.randint(111,888)
                        subject = f"{random.choice(dialogue.splendid)} Subject {subject_number}, what is your name and favorite monster?"
                        log_data(f"Dr. Brain | {subject}")
                        speak_and_print(subject)
                        #!loop 3
                        while answer is False:
                            name_monster = spk.transcribe()
                            if name_monster:
                                log_data(f"User | {name_monster}")
                                #!Sleep
                                if go_to_sleep(name_monster):
                                    answer = True
                                    break
                                speak_and_print(f"{random.choice(dialogue.pause)}")
                                nickname = "create a cool nickname which includes: " + name_monster
                                response = ad.llm_chat.invoke(char_prompt.format(query=nickname))
                                if response:
                                    log_data(f"Dr. Brain | {response.content}")
                                    speak_and_print(response.content)
                                    speak_and_print(f"One more subject for the Department of {random.choice(dialogue.department)}. Subject {subject_number}, {random.choice(dialogue.attend)} {random.choice(dialogue.classes)} class.")
                                    answer = True
                    elif "no" in decision.lower():
                        no_subject = f"{random.choice(dialogue.blast)} ... Astra Mor Technician, ask you're question already!"
                        log_data(f"Dr. Brain | {no_subject}")
                        speak_and_print(no_subject)
                        #!loop 3
                        while answer is False:
                            question = spk.transcribe()
                            if question:
                                log_data(f"User | {question}")
                                #!Sleep
                                if go_to_sleep(question):
                                    answer = True
                                    break
                                response = ad.llm_chat.invoke(char_prompt.format(query=question))
                                if response:
                                    log_data(f"Dr. Brain | {response.content}")
                                    speak_and_print(f"{random.choice(dialogue.pause)}")
                                    speak_and_print(response.content)
                                    answer = True
                    #!Sleep
                    elif go_to_sleep(decision):
                        answer = True
                        break
                    else:
                        speak_and_print("Please answer yes or no.")
    elif text and "exit" in text.lower():
        print("Exiting...")
        break


