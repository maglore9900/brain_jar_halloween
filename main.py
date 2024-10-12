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
        
while True:
    #loop 1
    answer = False
    text = spk.transcribe()
    if text and env("WAKE_WORD").lower() in text.lower() and env("CHARACTER").lower() in text.lower():
        if "wake" in text.lower():
            print("Waking up...")
            log_data(f"Dr. Brain | {text}")
            wake_up = f"{random.choice(dialogue.wake)} ... Astra Mor Technician, who approaches? ... A new test subject?"
            log_data(f"Dr. Brain | {wake_up}")
            print(wake_up)
            spk.stream(f"{wake_up}")
            #loop 2
            while answer is False:
                decision = spk.transcribe()
                if decision:
                    log_data(f"User | {decision}")
                    if "yes" in decision.lower():
                        subject = f"Test subject {random.randint(111,888)}, what is your name and favorite monster?"
                        log_data(f"Dr. Brain | {subject}")
                        print(subject)
                        spk.stream(f"{subject}")
                        #loop 3
                        while answer is False:
                            name_monster = spk.transcribe()
                            if name_monster:
                                log_data(f"User | {name_monster}")
                                if "back" in name_monster.lower() and "sleep" in name_monster.lower():
                                    answer = True
                                    go_sleep = f"{random.choice(dialogue.sleep)}"
                                    log_data(f"Dr. Brain | {go_sleep}")
                                    print(go_sleep)
                                    spk.stream(f"{go_sleep}")
                                    break
                                if env("SPEECH_ENABLED").lower() == "true":
                                    spk.stream(f"{random.choice(dialogue.pause)}")
                                nickname = "create a cool nickname which includes: " + name_monster
                                response = ad.llm_chat.invoke(char_prompt.format(query=nickname))
                                if response:
                                    log_data(f"Dr. Brain | {response.content}")
                                    print(response.content)
                                    if env("SPEECH_ENABLED").lower() == "true":
                                        spk.stream(response.content)
                                    answer = True
                    elif "no" in decision.lower():
                        no_subject = f"{random.choice(dialogue.blast)} ... Astra Mor Technician, ask you're question already!"
                        log_data(f"Dr. Brain | {no_subject}")
                        print(no_subject)
                        spk.stream(f"{no_subject}")
                        question = spk.transcribe()
                        if question:
                            log_data(f"User | {question}")
                            if "back" in question.lower() and "sleep" in question.lower():
                                answer = True
                                go_sleep = f"{random.choice(dialogue.sleep)}"
                                log_data(f"Dr. Brain | {go_sleep}")
                                print(go_sleep)
                                spk.stream(f"{go_sleep}")
                                break
                            #loop 3
                            while answer is False:
                                response = ad.llm_chat.invoke(char_prompt.format(query=question))
                                if response:
                                    log_data(f"Dr. Brain | {response.content}")
                                    print(response.content)
                                    if env("SPEECH_ENABLED").lower() == "true":
                                        spk.stream(f"{random.choice(dialogue.pause)}")
                                    if env("SPEECH_ENABLED").lower() == "true":
                                        spk.stream(response.content)
                                    answer = True
                    elif "back" in decision.lower() and "sleep" in decision.lower():
                        answer = True
                        go_sleep = f"{random.choice(dialogue.sleep)}"
                        log_data(f"Dr. Brain | {go_sleep}")
                        print(go_sleep)
                        spk.stream(f"{go_sleep}")
                        break
                    else:
                        spk.stream("Please answer yes or no.")
    elif text and "exit" in text.lower():
        print("Exiting...")
        break


