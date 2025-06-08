from flask import Flask, request, jsonify, send_from_directory
import os
import json
import logging
from music_composition import *

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API keys for services
API_KEYS = {
    "gemini": "AIzaSyCoLIPwyKC0Gg1WAdQLEYYgmIybFDNrwlI",
    "mistral": "jolgs3OzM3K93EBi1DNTF0itzEOQVaG7"
}

# Ensure output directories exist
os.makedirs("static/output", exist_ok=True)

# Color mapping for different moods
MOOD_COLORS = {
    "happy": "#FFD700",     # Gold
    "sad": "#4682B4",       # Steel Blue
    "energetic": "#FF4500", # Orange Red
    "calm": "#20B2AA",      # Light Sea Green
    "romantic": "#DA70D6",  # Orchid
    "mysterious": "#483D8B", # Dark Slate Blue
    "dark": "#36454F",      # Charcoal
    "bright": "#FF7F50",    # Coral
    "melancholic": "#778899", # Light Slate Gray
    "triumphant": "#B8860B", # Dark Goldenrod
    "playful": "#FF69B4",   # Hot Pink
    "nostalgic": "#9370DB", # Medium Purple
    # Default if no matching mood
    "default": "#6A5ACD"    # Slate Blue (default in frontend)
}

def determine_color_from_mood(mood):
    """Determine a color hex code based on the mood input"""
    mood_lower = mood.lower()
    
    for key, color in MOOD_COLORS.items():
        if key in mood_lower:
            return color
            
    return MOOD_COLORS["default"]

@app.route('/')
def index():
    """Serve the main index.html page"""
    return send_from_directory('.', 'templates/index.html')

@app.route('/static/output/')
def serve_output(filename):
    """Serve generated output files"""
    return send_from_directory('static/output', filename)

@app.route('/generate', methods=['POST'])
def generate_music():
    """Generate music based on form inputs"""
    try:
        # Extract form data
        title = request.form.get('title')
        style = request.form.get('style')
        mood = request.form.get('mood')
        constraints_text = request.form.get('constraints', '{}')
        
        logger.info(f"Received music generation request: {title}, {style}, {mood}")
        
        # Parse constraints (either JSON or plain text)
        try:
            constraints = json.loads(constraints_text)
        except json.JSONDecodeError:
            # If not valid JSON, treat as text description
            constraints = {"description": constraints_text}
            logger.info(f"Using text constraints: {constraints}")
        
        # Initialize the orchestrator
        orchestrator = CompositionOrchestrator(API_KEYS)
        
        # Generate the composition
        composition_audio = orchestrator.compose_music(
            style=style,
            mood=mood,
            title=title,
            constraints=constraints
        )
        
        # If no audio was generated, this is an error
        if not composition_audio or not os.path.exists(composition_audio):
            return jsonify({
                "error": "Failed to generate audio file",
                "color": determine_color_from_mood(mood)
            })
        
        # Determine MIDI path (the system might not actually generate a MIDI file)
        midi_path = composition_audio.replace(".wav", ".mid").replace(".mp3", ".mid")
        if not os.path.exists(midi_path):
            # If MIDI doesn't exist, use the same as audio for download
            midi_path = composition_audio.audio_path
        
        # Return success response with paths to generated files
        return jsonify({
            "title": "Song",
            "audio_path": os.path.basename(composition_audio),
            "midi_path": os.path.basename(midi_path),
            "color": determine_color_from_mood(mood)
        })
        
    except Exception as e:
        logger.error(f"Error generating music: {str(e)}", exc_info=True)
        return jsonify({
            "error": f"An error occurred: {str(e)}",
            "color": determine_color_from_mood(request.form.get('mood', ''))
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)