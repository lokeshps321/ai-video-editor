import subprocess

ass_content = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,100,&H00FFFFFF,&H000000FF,&H00000000,&HA0000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:05.00,Default,,0,0,0,,Hello World
"""
with open("/tmp/test_color.ass", "w") as f:
    f.write(ass_content)

subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=1920x1080:d=2", "-vf", "subtitles=/tmp/test_color.ass", "/tmp/out.mp4"], capture_output=True)
subprocess.run(["ffprobe", "-show_frames", "-select_streams", "v:0", "/tmp/out.mp4"], capture_output=True)
