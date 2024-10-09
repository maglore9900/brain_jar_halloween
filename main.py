from modules import adapter, speak_modified as speak
from prompts import prompts
import environ
import os

env = environ.Env()
environ.Env.read_env()

spk = speak.Speak(env)
ad = adapter.Adapter(env)
char = env("CHARACTER").lower()
char_prompt = getattr(prompts, char, "You are a helpful assistant.") + "\nAnswer the following request: {query}"

while True:
    text = spk.transcribe()
    if env("WAKE_WORD_ENABLED").lower() == "true":
        if text and env("WAKE_WORD").lower() in text.lower() and env("CHARACTER").lower() in text.lower():
            if "wake" in text.lower():
                print("Waking up...")
                spk.stream("Astra Mor Technician, who approaches? ... A new test subject?")
                while True:
                    decision = spk.transcribe()
                    if decision:
                        if "yes" in decision.lower():
                            spk.stream("Subject X, what is your name and favorite monster?")
                            while True:
                                print("nickname")
                                name_monster = spk.transcribe()
                                if name_monster:
                                    print(name_monster)
                                    nickname = "create a cool knickname which includes: " + name_monster
                                    response = ad.llm_chat.invoke(char_prompt.format(query=nickname))
                                    if response:
                                        print(response.content)
                                        if env("SPEECH_ENABLED").lower() == "true":
                                            spk.stream(response.content)
                        elif "no" in decision.lower():
                            pass
                        else:
                            spk.stream("Please answer yes or no.")
            if "exit" in text.lower():
                break

