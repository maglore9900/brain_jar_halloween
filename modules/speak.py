
import noisereduce as nr
import numpy as np
import pyaudio
import speech_recognition as sr
import pyttsx3
import os
import random
import urllib.parse
import requests
from pydub import AudioSegment
from pydub.effects import low_pass_filter
from collections import deque


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Speak:
    def __init__(self, env):
        self.url = env("STREAM_SPEAK_URL")
        self.microphone = sr.Microphone()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.model_name = env("LISTEN_MODEL".lower()) or "whisper"
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.noise_threshold = 500  # Initial placeholder for noise threshold
        self.recent_noise_levels = deque(maxlen=30)  # Track recent noise levels for dynamic adjustment
        self.voice = env("ALL_TALK_VOICE")
        self.silence = float(env("TIME_SILENCE"))

        # Initialize transcription models
        if self.model_name == "whisper":
            from faster_whisper import WhisperModel
            self.whisper_model_path = "large-v2"
            self.whisper_model = WhisperModel(self.whisper_model_path, device="cuda")  # Nvidia GPU mode
        else:
            self.recognizer = sr.Recognizer()

    def adjust_noise_threshold(self, audio_chunk):
        """Dynamically adjust the noise threshold based on the ambient noise levels of the current chunk."""
        noise_level = np.abs(audio_chunk).mean()
        self.recent_noise_levels.append(noise_level)
        
        # Calculate a new threshold based on recent noise levels (running average)
        self.noise_threshold = np.mean(self.recent_noise_levels)

    def listen_to_microphone(self):
        """Function to listen to the microphone input and return raw audio data after applying dynamic noise reduction."""
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True, frames_per_buffer=self.chunk_size)
        stream.start_stream()
        print("Listening...")

        audio_data = b""
        silence_duration = self.silence  # Time of silence in seconds before stopping
        silence_counter = 0
        detected_speech = False
        
        while True:
            data = stream.read(self.chunk_size)
            audio_data += data

            # Convert to numpy array for noise reduction and dynamic adjustment
            np_data = np.frombuffer(data, dtype=np.int16)
            
            # Adjust noise threshold dynamically using the current chunk
            self.adjust_noise_threshold(np_data)
            
            # Reduce noise in the current chunk
            reduced_noise_data = nr.reduce_noise(y=np_data, sr=self.sample_rate)
            
            # Check if speech is detected based on the dynamically adjusted noise threshold
            if np.abs(reduced_noise_data).mean() > self.noise_threshold:
                detected_speech = True
                silence_counter = 0  # Reset silence counter when speech is detected
            elif detected_speech:  # If we already detected speech and now there is silence
                silence_counter += self.chunk_size / self.sample_rate
                if silence_counter >= silence_duration:
                    print("Silence detected. Stopping.")
                    break
        
        stream.stop_stream()
        stream.close()
        p.terminate()

        return audio_data

    def transcribe(self):
        """
        Function to transcribe audio from the microphone. Stops when no speech is detected.
        """
        print("Listening until silence is detected.")

        audio_data = self.listen_to_microphone()

        # Transcription logic here
        if self.model_name == "whisper":
            energy_threshold = 0.0001
            audio_np = np.frombuffer(audio_data, np.int16).astype(np.float32) / 32768.0
            energy = np.mean(np.abs(audio_np))
            if energy > energy_threshold:
                segments, _ = self.whisper_model.transcribe(audio_np, beam_size=5)
                transcription = " ".join([segment.text for segment in segments])
                print(f"Whisper Transcription: {transcription}")
                return transcription
        else:
            with self.microphone as source:
                try:
                    audio = sr.AudioData(audio_data, self.sample_rate, 2)
                    transcription = self.recognizer.recognize_google(audio)
                    print(f"Google Transcription: {transcription}")
                    return transcription
                except:
                    pass
                
    def stream(self, text):
        # Example parameters
        voice = self.voice
        language = "en"
        output_file = "stream_output.wav"
        
        # Encode the text for URL
        encoded_text = urllib.parse.quote(text)
        
        # Create the streaming URL
        streaming_url = f"http://localhost:7851/api/tts-generate-streaming?text={encoded_text}&voice={voice}&language={language}&output_file={output_file}"
        
        
        try:
            # Stream the audio data
            response = requests.get(streaming_url, stream=True)
            
            # Initialize PyAudio
            p = pyaudio.PyAudio()
            stream = None
            
            # Process the audio stream in chunks
            chunk_size = 1024 * 6  # Adjust chunk size if needed
            audio_buffer = b''

            for chunk in response.iter_content(chunk_size=chunk_size):
                audio_buffer += chunk

                if len(audio_buffer) < chunk_size:
                    continue
                
                audio_segment = AudioSegment(
                    data=audio_buffer,
                    sample_width=2,  # 2 bytes for 16-bit audio
                    frame_rate=24000,  # Assumed frame rate, adjust as necessary
                    channels=1  # Assuming mono audio
                )

                if stream is None:
                    # Define stream parameters without any modifications
                    stream = p.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=audio_segment.frame_rate,
                                    output=True)

                # Play the original chunk (without any modification)
                stream.write(audio_segment.raw_data)

                # Reset buffer
                audio_buffer = b''

            # Final cleanup
            if stream:
                stream.stop_stream()
                stream.close()
            p.terminate()
            
        except:
            self.engine.say(text)
            self.engine.runAndWait()
            
    def apply_vinyl_effect(self, audio, rate):
        """
        Apply wow (slow pitch variation), tape warble (both static-like and pitch variation), crackle noise, 
        and a subtle low-pass filter to the audio.
        """
        crackle_intensity = 2  # Crackle noise intensity
        crackle_prob = 0.9  # High probability for crackle

        wow_intensity = 10  # Wow effect (slow pitch modulations)
        warble_intensity = 0.02  # Pitch warble (irregular speed variations)
        static_intensity = 10  # Static-like warble effect

        # Generate wow effect (slow pitch modulations)
        wow = np.sin(np.linspace(0, 2 * np.pi, len(audio))) * wow_intensity

        # Create a copy of the audio to apply wow, warble, and crackle
        audio_with_effects = np.copy(audio)
        for i in range(1, len(audio_with_effects)):
            # Apply wow (slow pitch modulation)
            audio_with_effects[i] = audio_with_effects[i] + wow[i]

            # Apply pitch-based warble (irregular pitch variations)
            if np.random.rand() < 0.5:  # Medium probability for warble
                audio_with_effects[i] = audio_with_effects[i] * (1 + np.random.normal(0, warble_intensity))

            # Apply static-like warble (adds subtle static noise)
            if np.random.rand() < 1:  # Low probability for static-like warble
                audio_with_effects[i] = audio_with_effects[i] + np.random.normal(0, static_intensity)

            # Apply crackle randomly
            if np.random.rand() < crackle_prob:
                audio_with_effects[i] = audio_with_effects[i] + np.random.normal(0, crackle_intensity)

        # Convert NumPy array back to AudioSegment for applying the low-pass filter
        segment = AudioSegment(
            audio_with_effects.tobytes(),  # Convert NumPy array to raw bytes
            frame_rate=rate,
            sample_width=2,  # 16-bit audio = 2 bytes
            channels=1       # Mono audio
        )

        # Apply a gentle low-pass filter to reduce higher frequencies
        filtered_segment = low_pass_filter(segment, cutoff=5000)  # Gentle cutoff at 6000 Hz

        # Return the processed audio back as a NumPy array
        return np.frombuffer(filtered_segment.raw_data, dtype=np.int16) 
        
    # def vox(self, text):
    #     """
    #     Modified stream function that receives streaming audio and applies vinyl-like effects in real-time.
    #     """
    #     # Example parameters
    #     voice = self.voice
    #     language = "en"
    #     output_file = "stream_output.wav"

    #     # Encode the text for URL
    #     encoded_text = urllib.parse.quote(text)

    #     # Create the streaming URL
    #     streaming_url = f"http://localhost:7851/api/tts-generate-streaming?text={encoded_text}&voice={voice}&language={language}&output_file={output_file}"
        
    #     spooky = AudioSegment.from_file("spooky2.mp3", format="mp3")
    #     spooky_len = len(spooky)
    #     spooky = spooky.set_frame_rate(24000).set_channels(1).set_sample_width(2)
    #     spooky = spooky.fade_in(2000)
        
    #     # Split spooky audio into chunks
        
    #     chunk_index = 0  # Initialize chunk index
        
    #     try:
    #         # Stream the audio data
    #         response = requests.get(streaming_url, stream=True)

    #         # Initialize PyAudio
    #         p = pyaudio.PyAudio()
    #         stream = None

    #         # Process the audio stream in chunks
    #         chunk_size = 1024 * 6  # Adjust chunk size if needed
    #         audio_buffer = b''

    #         for chunk in response.iter_content(chunk_size=chunk_size):
    #             audio_buffer += chunk

    #             if len(audio_buffer) < chunk_size:
    #                 continue

    #             # Convert the buffer to an AudioSegment for processing
    #             audio_segment = AudioSegment(
    #                 data=audio_buffer,
    #                 sample_width=2,  # 2 bytes for 16-bit audio
    #                 frame_rate=24000,  # Assumed frame rate, adjust as necessary
    #                 channels=1  # Assuming mono audio
    #             )

    #             # Apply the vinyl effect to the audio segment
    #             audio_np = np.frombuffer(audio_segment.raw_data, dtype=np.int16)

    #             # Ensure we're applying effects in a subtle manner
    #             # vinyl_audio = self.apply_vinyl_effect(audio_np, audio_segment.frame_rate)
    #             spooky_chunks = self.apply_spooky_overlay_incrementally(spooky, chunk_index, spooky_len, chunk_size)
    #             vinyl_audio = self.apply_vinyl_effect_with_spooky(audio_np, audio_segment.frame_rate, spooky_chunks, chunk_index)
    #             # Convert the modified audio back to bytes
    #             vinyl_audio_segment = AudioSegment(
    #                 vinyl_audio.tobytes(),
    #                 sample_width=2,
    #                 frame_rate=audio_segment.frame_rate,
    #                 channels=1
    #             )

    #             if stream is None:
    #                 # Define stream parameters
    #                 stream = p.open(format=pyaudio.paInt16,
    #                                 channels=1,
    #                                 rate=vinyl_audio_segment.frame_rate,
    #                                 output=True)

    #             # Play the modified chunk with the vinyl effect
    #             stream.write(vinyl_audio_segment.raw_data)

    #             # Reset buffer
    #             audio_buffer = b''
    #             chunk_index += 1

    #         # Final cleanup
    #         if stream:
    #             stream.stop_stream()
    #             stream.close()
    #         p.terminate()

    #     except Exception as e:
    #         print(f"An error occurred during streaming: {str(e)}")
    #         self.engine.say(text)
    #         self.engine.runAndWait()

       

    # def vox(self, text):
    #     # Example parameters
    #     voice = self.voice
    #     language = "en"
    #     output_file = "stream_output.wav"
        
    #     # Load spooky sound
    #     spooky = AudioSegment.from_mp3("spooky.mp3")
    #     spooky_len = len(spooky)
        
    #     # Encode the text for URL
    #     encoded_text = urllib.parse.quote(text)
        
    #     # Create the streaming URL
    #     streaming_url = f"http://localhost:7851/api/tts-generate-streaming?text={encoded_text}&voice={voice}&language={language}&output_file={output_file}"
        
    #     try:
    #         # Stream the audio data
    #         response = requests.get(streaming_url, stream=True)
            
    #         # Initialize PyAudio
    #         p = pyaudio.PyAudio()
    #         stream = None
            
    #         # Process the audio stream in chunks
    #         chunk_size = 1024 * 6  # Adjust chunk size if needed
    #         audio_buffer = b''
    #         chunk_index = 0
            
    #         for chunk in response.iter_content(chunk_size=chunk_size):
    #             audio_buffer += chunk

    #             if len(audio_buffer) < chunk_size:
    #                 continue
                
    #             audio_segment = AudioSegment(
    #                 data=audio_buffer,
    #                 sample_width=2,  # 2 bytes for 16-bit audio
    #                 frame_rate=24000,  # Assumed frame rate, adjust as necessary
    #                 channels=1  # Assuming mono audio
    #             )

    #             if stream is None:
    #                 # Define stream parameters without any modifications
    #                 stream = p.open(format=pyaudio.paInt16,
    #                                 channels=1,
    #                                 rate=audio_segment.frame_rate,
    #                                 output=True)

    #             # Apply the spooky overlay incrementally
    #             spooky_chunks = self.apply_spooky_overlay_incrementally(spooky, chunk_index, spooky_len, len(audio_segment))
    #             spooky_overlay = spooky_chunks[chunk_index % len(spooky_chunks)]  # Ensure the overlay loops
                
    #             # Reduce the volume of the spooky audio (so the streaming audio is still dominant)
    #             # spooky_overlay = spooky_overlay - 40  # Lower spooky volume by 15 dB
                
    #             # Mix the spooky audio with the current audio segment
    #             combined = audio_segment.overlay(spooky_overlay, position=0)
                
    #             # Play the combined audio (stream + spooky overlay)
    #             stream.write(combined.raw_data)
    #             # stream.write(audio_segment.raw_data)

    #             # Reset buffer and increment chunk index
    #             audio_buffer = b''
    #             chunk_index += 1

    #         # Final cleanup
    #         if stream:
    #             stream.stop_stream()
    #             stream.close()
    #         p.terminate()
            
    #     except:
    #         self.engine.say(text)
    #         self.engine.runAndWait()


    # def apply_spooky_overlay_incrementally(self, spooky, chunk_index, spooky_len, chunk):
    #     """
    #     Split spooky audio into chunks and start at a random position, looping the audio if necessary.
    #     """
    #     spooky_chunks = []
    #     # Start the spooky sound from a random position
    #     if chunk_index == 0:
    #         start_position = random.randint(0, max(0, spooky_len - chunk))
    #         print(f"Starting spooky sound at position: {start_position}")
    #         spooky = spooky[start_position:]  # Trim the spooky sound starting at a random position

    #     # Split the spooky audio into chunks
    #     for i in range(0, len(spooky), chunk):
    #         spooky_chunk = spooky[i:i+chunk]
    #         # spooky_chunk = spooky_chunk + 20  # Increase volume by 20 dB before lowering it when overlaid
    #         spooky_chunks.append(spooky_chunk)
        
    #     return spooky_chunks




    def vox(self, text):
        # Example parameters
        voice = self.voice
        language = "en"
        output_file = "stream_output.wav"
        
        # Load spooky sound
        spooky = AudioSegment.from_mp3("spooky2.mp3")
        spooky_len = len(spooky)
        
        # Encode the text for URL
        encoded_text = urllib.parse.quote(text)
        
        # Create the streaming URL
        streaming_url = f"http://localhost:7851/api/tts-generate-streaming?text={encoded_text}&voice={voice}&language={language}&output_file={output_file}"
        
        try:
            # Stream the audio data
            response = requests.get(streaming_url, stream=True)
            
            # Initialize PyAudio
            p = pyaudio.PyAudio()
            stream = None
            
            # Process the audio stream in chunks
            chunk_size = 1024 * 6  # Adjust chunk size if needed
            audio_buffer = b''
            chunk_index = 0
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                audio_buffer += chunk

                if len(audio_buffer) < chunk_size:
                    continue
                
                audio_segment = AudioSegment(
                    data=audio_buffer,
                    sample_width=2,  # 2 bytes for 16-bit audio
                    frame_rate=24000,  # Assumed frame rate, adjust as necessary
                    channels=1  # Assuming mono audio
                )
                audio_segment = audio_segment.set_channels(2)
                whisper_audio = self.apply_whisper_effect2(audio_segment)
                if stream is None:
                    # Define stream parameters without any modifications
                    stream = p.open(format=pyaudio.paInt16,
                                    channels=2,
                                    rate=audio_segment.frame_rate,
                                    output=True)

                # Apply the spooky overlay incrementally
                # spooky_chunks = self.apply_spooky_overlay_incrementally(spooky, chunk_index, spooky_len, len(audio_segment))
                spooky_chunks = self.apply_spooky_overlay_incrementally(spooky, chunk_index, spooky_len, len(audio_segment), fade_in_duration=1000, fade_out_duration=1000)
                spooky_overlay = spooky_chunks[chunk_index % len(spooky_chunks)]  # Ensure the overlay loops
                spooky_overlay = spooky_overlay.apply_gain(-10)  # Lower spooky volume by 20 dB
                spooky_overlay = spooky_overlay.high_pass_filter(1500)  # Apply a high-pass filter to the spooky overlay
                # Manually mix the audio streams by combining the raw audio samples
                # combined = self.mix_audio_segments(audio_segment, spooky_overlay)
                combined = self.mix_audio_segments(whisper_audio, spooky_overlay)
                # Play the combined audio (stream + spooky overlay)
                stream.write(combined.raw_data)

                # Reset buffer and increment chunk index
                audio_buffer = b''
                chunk_index += 1

            # Final cleanup
            if stream:
                stream.stop_stream()
                stream.close()
            p.terminate()
            
        except Exception as e:
            print(f"An error occurred during streaming: {str(e)}")
            self.engine.say(text)
            self.engine.runAndWait()

    def apply_spooky_overlay_incrementally(self, spooky, chunk_index, spooky_len, chunk, fade_in_duration=1000, fade_out_duration=1000):
        """
        Split spooky audio into chunks, apply fade-in and fade-out effects, and start at a random position.
        """
        spooky_chunks = []
        
        # Start the spooky sound from a random position
        if chunk_index == 0:
            start_position = random.randint(0, max(0, spooky_len - chunk))
            spooky = spooky[start_position:]  # Trim the spooky sound starting at a random position

        # Apply fade-in to the first chunk
        if chunk_index == 0 and fade_in_duration > 0:
            spooky = spooky.fade_in(fade_in_duration)
        
        # Split the spooky audio into chunks
        for i in range(0, len(spooky), chunk):
            spooky_chunk = spooky[i:i+chunk]

            # Apply fade-out to the last chunk
            if i + chunk >= len(spooky) and fade_out_duration > 0:
                spooky_chunk = spooky_chunk.fade_out(fade_out_duration)

            spooky_chunks.append(spooky_chunk)
        
        return spooky_chunks


    # def mix_audio_segments(self, audio_segment, spooky_segment, spooky_boost_factor=0.8):
    #     """
    #     Manually combine two audio segments by mixing their samples, ensuring they have the same frame rate and length.
    #     Apply a volume boost to the spooky segment.
    #     """
    #     # Ensure both audio segments have the same frame rate (resample if necessary)
    #     if audio_segment.frame_rate != spooky_segment.frame_rate:
    #         spooky_segment = spooky_segment.set_frame_rate(audio_segment.frame_rate)

    #     # Convert both segments to raw data (sample arrays)
    #     audio_samples = np.frombuffer(audio_segment.raw_data, dtype=np.int16)
    #     spooky_samples = np.frombuffer(spooky_segment.raw_data, dtype=np.int16)

    #     # Ensure both audio samples are the same length
    #     min_length = min(len(audio_samples), len(spooky_samples))
    #     audio_samples = audio_samples[:min_length]
    #     spooky_samples = spooky_samples[:min_length]

    #     # Apply the volume boost to the spooky samples
    #     spooky_samples = spooky_samples.astype(np.float32) * spooky_boost_factor

    #     # Mix the two audio samples by averaging them (or adjust for desired balance)
    #     mixed_samples = (audio_samples.astype(np.float32) + spooky_samples) / 2
    #     mixed_samples = np.clip(mixed_samples, -32768, 32767).astype(np.int16)  # Ensure no overflow for 16-bit audio

    #     # Convert mixed samples back to raw audio data
    #     mixed_audio = mixed_samples.tobytes()

    #     # Create a new AudioSegment with the mixed audio
    #     combined_segment = AudioSegment(
    #         data=mixed_audio,
    #         sample_width=2,  # 2 bytes for 16-bit audio
    #         frame_rate=audio_segment.frame_rate,
    #         channels=1
    #     )
        
    #     return combined_segment
    def mix_audio_segments(self, audio_segment, spooky_segment, spooky_boost_factor=0.8):
        """
        Manually combine two audio segments by mixing their samples, ensuring they have the same frame rate and length.
        Apply a volume boost to the spooky segment.
        """
        # Ensure both audio segments have the same frame rate (resample if necessary)
        if audio_segment.frame_rate != spooky_segment.frame_rate:
            spooky_segment = spooky_segment.set_frame_rate(audio_segment.frame_rate)

        # Apply the volume boost to the spooky segment using pydub's apply_gain
        spooky_segment = spooky_segment.apply_gain(20 * spooky_boost_factor)  # Adjust gain for the boost factor

        # Ensure both segments are the same length
        if len(spooky_segment) < len(audio_segment):
            spooky_segment = spooky_segment + AudioSegment.silent(duration=len(audio_segment) - len(spooky_segment))
        elif len(audio_segment) < len(spooky_segment):
            audio_segment = audio_segment + AudioSegment.silent(duration=len(spooky_segment) - len(audio_segment))

        # Use pydub's overlay method to combine both audio segments
        combined_segment = audio_segment.overlay(spooky_segment)

        return combined_segment


    def apply_whisper_effect(self, audio_segment, white_noise_gain=-45, speech_gain=2, noise_layers=2):
        """
        Modify the audio segment to sound like a whisper.
        - Apply a high-pass filter to remove low frequencies.
        - Apply a low-pass filter to keep higher frequencies.
        - Add subtle noise to simulate breathiness with adjustable intensity.
        - Ensure timing, frame rates, and gain are handled properly to avoid slowing down.
        """
        from pydub.generators import WhiteNoise

        # Apply a high-pass filter to remove low-end frequencies (e.g., below 500 Hz)
        whisper_audio = audio_segment.high_pass_filter(1500)

        # Apply a low-pass filter to cut off higher frequencies (e.g., above 6000 Hz)
        whisper_audio = whisper_audio.low_pass_filter(4000)

        # Manually adjust the volume of the speech (instead of normalizing)
        whisper_audio = whisper_audio.apply_gain(speech_gain)

        # Generate white noise to simulate the breathy aspect of a whisper
        noise = WhiteNoise().to_audio_segment(duration=len(whisper_audio)).set_frame_rate(whisper_audio.frame_rate).apply_gain(white_noise_gain)

        # Ensure that the noise and speech are the same length and frame rate
        if len(noise) != len(whisper_audio):
            noise = noise + AudioSegment.silent(duration=len(whisper_audio) - len(noise))

        # If more intensity is desired, layer additional noise on top
        for _ in range(noise_layers - 1):
            extra_noise = WhiteNoise().to_audio_segment(duration=len(whisper_audio)).set_frame_rate(whisper_audio.frame_rate).apply_gain(white_noise_gain)
            noise = noise.overlay(extra_noise)

        # Overlay the noise with the whisper audio
        whisper_audio = whisper_audio.overlay(noise, position=0)

        return whisper_audio
    
    def apply_whisper_effect2(self, audio_segment, white_noise_gain=-45, speech_gain=2, pitch_factor=0.9, noise_layers=2):
        """
        Modify the audio segment to sound like a whisper and adjust the pitch.
        - Apply a high-pass filter to remove low frequencies.
        - Apply a low-pass filter to keep higher frequencies.
        - Add subtle noise to simulate breathiness with adjustable intensity.
        - Adjust pitch by resampling the audio.
        """
        from pydub.generators import WhiteNoise
        # Adjust the pitch of the original audio by changing the playback speed
        new_frame_rate = int(audio_segment.frame_rate * pitch_factor)
        pitched_audio = audio_segment._spawn(audio_segment.raw_data, overrides={'frame_rate': new_frame_rate})
        
        # Maintain original frame rate (so the duration stays the same after pitch change)
        pitched_audio = pitched_audio.set_frame_rate(audio_segment.frame_rate)

        # Apply a high-pass filter to remove low-end frequencies (e.g., below 500 Hz)
        whisper_audio = pitched_audio.high_pass_filter(500)

        # Apply a low-pass filter to cut off higher frequencies (e.g., above 6000 Hz)
        whisper_audio = whisper_audio.low_pass_filter(6000)

        # Manually adjust the volume of the speech (instead of normalizing)
        whisper_audio = whisper_audio.apply_gain(speech_gain)

        # Generate white noise to simulate the breathy aspect of a whisper
        noise = WhiteNoise().to_audio_segment(duration=len(whisper_audio)).set_frame_rate(whisper_audio.frame_rate).apply_gain(white_noise_gain)

        # Ensure that the noise and speech are the same length and frame rate
        if len(noise) != len(whisper_audio):
            noise = noise + AudioSegment.silent(duration=len(whisper_audio) - len(noise))

        # If more intensity is desired, layer additional noise on top
        for _ in range(noise_layers - 1):
            extra_noise = WhiteNoise().to_audio_segment(duration=len(whisper_audio)).set_frame_rate(whisper_audio.frame_rate).apply_gain(white_noise_gain)
            noise = noise.overlay(extra_noise)

        # Overlay the noise with the whisper audio
        whisper_audio = whisper_audio.overlay(noise, position=0)

        return whisper_audio

    


    # def vox(self, text):
    #     # Example parameters
    #     voice = self.voice
    #     language = "en"
    #     output_file = "stream_output.wav"
        
    #     # Load spooky sound
    #     spooky = AudioSegment.from_mp3("spooky.mp3")
    #     spooky_len = len(spooky)
        
    #     # Encode the text for URL
    #     encoded_text = urllib.parse.quote(text)
        
    #     # Create the streaming URL
    #     streaming_url = f"http://localhost:7851/api/tts-generate-streaming?text={encoded_text}&voice={voice}&language={language}&output_file={output_file}"
        
    #     try:
    #         # Stream the audio data
    #         response = requests.get(streaming_url, stream=True)
            
    #         # Initialize PyAudio
    #         p = pyaudio.PyAudio()
    #         stream = None
            
    #         # Process the audio stream in chunks
    #         chunk_size = 1024 * 6  # Adjust chunk size if needed
    #         audio_buffer = b''
    #         chunk_index = 0
            
    #         for chunk in response.iter_content(chunk_size=chunk_size):
    #             audio_buffer += chunk

    #             if len(audio_buffer) < chunk_size:
    #                 continue
                
    #             audio_segment = AudioSegment(
    #                 data=audio_buffer,
    #                 sample_width=2,  # 2 bytes for 16-bit audio
    #                 frame_rate=24000,  # Assumed frame rate, adjust as necessary
    #                 channels=1  # Assuming mono audio
    #             )
    #             audio_segment = audio_segment.set_channels(2)
    #             # Apply the whisper effect to the streamed audio
    #             whisper_audio = self.apply_whisper_effect(audio_segment)

    #             if stream is None:
    #                 # Define stream parameters, now with whisper effect applied
    #                 stream = p.open(format=pyaudio.paInt16,
    #                                 channels=2,  # Mono audio
    #                                 rate=whisper_audio.frame_rate,
    #                                 output=True)

    #             # Apply the spooky overlay incrementally
    #             spooky_chunks = self.apply_spooky_overlay_incrementally(spooky, chunk_index, spooky_len, len(whisper_audio))
    #             spooky_overlay = spooky_chunks[chunk_index % len(spooky_chunks)]  # Ensure the overlay loops
    #             spooky_overlay = spooky_overlay.apply_gain(-20)  # Lower spooky volume by 20 dB
    #             # Manually mix the audio streams by combining the raw audio samples
    #             combined = self.mix_audio_segments(whisper_audio, spooky_overlay)
                
    #             # Play the combined audio (stream + spooky overlay)
    #             stream.write(combined.raw_data)

    #             # Reset buffer and increment chunk index
    #             audio_buffer = b''
    #             chunk_index += 1

    #         # Final cleanup
    #         if stream:
    #             stream.stop_stream()
    #             stream.close()
    #         p.terminate()
            
    #     except Exception as e:
    #         print(f"An error occurred during streaming: {str(e)}")
    #         self.engine.say(text)
    #         self.engine.runAndWait()