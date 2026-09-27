import edge_tts

async def create_voice(text, output_file):
    communicate = edge_tts.Communicate(
        text,
        "en-US-AriaNeural"
    )

    await communicate.save(output_file)