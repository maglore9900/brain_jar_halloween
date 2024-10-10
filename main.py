from modules import adapter, speak
from prompts import prompts, dialogue
import environ
import random


env = environ.Env()
environ.Env.read_env()

spk = speak.Speak(env)
ad = adapter.Adapter(env)
char = env("CHARACTER").lower()
char_prompt = getattr(prompts, char, "You are a helpful assistant.") + "\nAnswer the following request: {query}"


while True:
    answer = False
    text = spk.transcribe()
    if env("WAKE_WORD_ENABLED").lower() == "true":
        if text and env("WAKE_WORD").lower() in text.lower() and env("CHARACTER").lower() in text.lower():
            if "wake" in text.lower():
                print("Waking up...")
                # spk.stream(random.choice(dialogue.wake))
                spk.stream(f"{random.choice(dialogue.wake)} ... Astra Mor Technician, who approaches? ... A new test subject?")
                while answer is False:
                    decision = spk.transcribe()
                    if decision:
                        if "yes" in decision.lower():
                            spk.stream(f"Test subject {random.randint(111,888)}, what is your name and favorite monster?")
                            while answer is False:
                                name_monster = spk.transcribe()
                                if name_monster:
                                    spk.stream(random.choice(dialogue.pause))
                                    print(name_monster)
                                    nickname = "create a cool nickname which includes: " + name_monster
                                    response = ad.llm_chat.invoke(char_prompt.format(query=nickname))
                                    if response:
                                        print(response.content)
                                        if env("SPEECH_ENABLED").lower() == "true":
                                            spk.stream(response.content)
                                        answer = True
                        elif "no" in decision.lower():
                            spk.stream("Astra Mor Technician, ask you're question already!")
                            question = spk.transcribe()
                            if question:
                                response = ad.llm_chat.invoke(char_prompt.format(query=question))
                                if response:
                                    spk.stream(random.choice(dialogue.pause))
                                    print(response.content)
                                    if env("SPEECH_ENABLED").lower() == "true":
                                        spk.stream(response.content)
                                    answer = True
                        else:
                            spk.stream("Please answer yes or no.")
        elif text and "exit" in text.lower():
            break


