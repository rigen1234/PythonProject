from moviepy import VideoFileClip

# Load video file
video = VideoFileClip("input_video.mp4")

# Extract audio
audio = video.audio

# Save audio to file
audio.write_audiofile("output_audio.mp3")

# Close resources
audio.close()
video.close()

print("✅ Audio extracted successfully!")