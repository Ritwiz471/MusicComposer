#!/usr/bin/env python3
"""
AI Music Composition Collaboration System (Enhanced Version - With Beats and Audio)

Improvements:
1. Melody generation now explains constraints clearly.
2. Tool descriptions now provide detailed instructions on the use of constraints.
3. Arrangement generation provides default dynamics if missing.
4. Final composition JSON extraction is robust.
5. Rhythm generation now includes a separate beat pattern for added variety.
6. An audio generation tool converts the Music21 score to WAV or MP3 using FluidSynth and ffmpeg.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from enum import Enum
import time
import re
import subprocess

# Music notation and playback
import music21
from music21 import midi

# LlamaIndex imports (adjust paths or imports as needed)
from llama_index.llms.gemini import Gemini
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums for musical consistency
class Scale(Enum):
    MAJOR = "major"
    MINOR = "minor"
    DORIAN = "dorian"
    MIXOLYDIAN = "mixolydian"
    PENTATONIC = "pentatonic"

class MusicalStyle(Enum):
    CLASSICAL = "classical"
    JAZZ = "jazz"
    POP = "pop"
    FOLK = "folk"
    ELECTRONIC = "electronic"

# Data structures for musical composition
class MelodyIdea:
    def __init__(self, notes: str, scale: Scale, tempo: int):
        self.notes = notes
        self.scale = scale
        self.tempo = tempo
    
    def to_dict(self):
        return {"notes": self.notes, "scale": self.scale.value, "tempo": self.tempo}
    
    @classmethod
    def from_dict(cls, data):
        return cls(notes=data["notes"], scale=Scale(data["scale"]), tempo=data["tempo"])

class HarmonyProgression:
    def __init__(self, chords: str, key: str):
        self.chords = chords
        self.key = key
    
    def to_dict(self):
        return {"chords": self.chords, "key": self.key}
    
    @classmethod
    def from_dict(cls, data):
        return cls(chords=data["chords"], key=data["key"])

# Updated RhythmPattern with a separate beats attribute
class RhythmPattern:
    def __init__(self, pattern: str, time_signature: str, groove: str, beats: Optional[str] = None):
        self.pattern = pattern
        self.time_signature = time_signature
        self.groove = groove
        self.beats = beats  # e.g., "B S B S" where B=kick and S=snare
    
    def to_dict(self):
        return {
            "pattern": self.pattern, 
            "time_signature": self.time_signature, 
            "groove": self.groove,
            "beats": self.beats
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            pattern=data["pattern"], 
            time_signature=data["time_signature"], 
            groove=data["groove"],
            beats=data.get("beats")
        )

class Arrangement:
    def __init__(self, instrumentation: List[str], structure: str, dynamics: str):
        self.instrumentation = instrumentation
        self.structure = structure
        self.dynamics = dynamics
    
    def to_dict(self):
        return {"instrumentation": self.instrumentation, "structure": self.structure, "dynamics": self.dynamics}
    
    @classmethod
    def from_dict(cls, data):
        return cls(instrumentation=data["instrumentation"], structure=data["structure"], dynamics=data["dynamics"])

class Composition:
    def __init__(self, 
                 title: str, 
                 melody: Optional[MelodyIdea] = None,
                 harmony: Optional[HarmonyProgression] = None,
                 rhythm: Optional[RhythmPattern] = None,
                 arrangement: Optional[Arrangement] = None,
                 score: Optional[str] = None,
                 audio_path: Optional[str] = None):
        self.title = title
        self.melody = melody
        self.harmony = harmony
        self.rhythm = rhythm
        self.arrangement = arrangement
        self.score = score  # Music21 score serialized to MusicXML
        self.audio_path = audio_path  # Path to the generated audio file (WAV or MP3)
        self.feedback = []
        self.iteration = 0
    
    def to_dict(self):
        return {
            "title": self.title,
            "melody": self.melody.to_dict() if self.melody else None,
            "harmony": self.harmony.to_dict() if self.harmony else None,
            "rhythm": self.rhythm.to_dict() if self.rhythm else None,
            "arrangement": self.arrangement.to_dict() if self.arrangement else None,
            "score": self.score,
            "audio_path": self.audio_path,
            "feedback": self.feedback,
            "iteration": self.iteration
        }
    
    @classmethod
    def from_dict(cls, data):
        comp = cls(title=data["title"])
        if data.get("melody"):
            comp.melody = MelodyIdea.from_dict(data["melody"])
        if data.get("harmony"):
            comp.harmony = HarmonyProgression.from_dict(data["harmony"])
        if data.get("rhythm"):
            comp.rhythm = RhythmPattern.from_dict(data["rhythm"])
        if data.get("arrangement"):
            comp.arrangement = Arrangement.from_dict(data["arrangement"])
        comp.score = data.get("score")
        comp.audio_path = data.get("audio_path")
        comp.feedback = data.get("feedback", [])
        comp.iteration = data.get("iteration", 0)
        return comp

# LLM Agent base class
class MusicAgent:
    def __init__(self, name: str, llm, system_prompt: str):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
        
    def generate_response(self, prompt: str) -> str:
        full_prompt = f"{self.system_prompt}\n\n{prompt}"
        self.memory.put(full_prompt)
        time.sleep(2)
        response = self.llm.complete(full_prompt)
        self.memory.put(response.text)
        logger.info(f"Agent {self.name} generated response")
        return response.text

# Specialized music agents
class MelodyAgent(MusicAgent):
    def __init__(self, llm):
        system_prompt = (
            "You are a Melody Agent specializing in creating compelling melodic themes. "
            "When provided with constraints, please respect the following:\n"
            "- max_length: Maximum number of notes or phrases allowed.\n"
            "- phrase_structure: The overall structure (e.g., 'ABAC') dictating repeated and contrasting sections.\n"
            "- contour: Desired overall shape (e.g., 'arch' means rise then fall in pitch).\n"
            "- harmony_emphasis: If true, emphasize chord tones in the melody to support later harmony generation.\n\n"
            "Return your answer in this format:\n"
            "NOTES: <space-separated note names (with optional durations)>\n"
            "SCALE: <scale type (minor, major, etc.)>\n"
            "TEMPO: <tempo in BPM>"
        )
        super().__init__("Melody Agent", llm, system_prompt)
    
    def create_melody(self, style: MusicalStyle, mood: str, constraints: Dict[str, Any]) -> MelodyIdea:
        prompt = f"""
        Create a new melody with:
        - Style: {style.value}
        - Mood: {mood}
        - Constraints: {constraints}
        
        Please ensure the melody adheres to the following guidelines:
        - Do not exceed {constraints.get('max_length', 'an unspecified number of')} notes.
        - Use a phrase structure such as {constraints.get('phrase_structure', 'default')}.
        - Follow an overall {constraints.get('contour', 'unspecified')} contour.
        - Emphasize chord tones if 'harmony_emphasis' is true.
        
        Format your response exactly as:
        NOTES: F4 G4 A4 ...
        SCALE: minor or major, etc.
        TEMPO: 90 BPM
        """
        response = self.generate_response(prompt)
        
        notes, scale, tempo = None, None, None
        
        for line in response.split('\n'):
            if line.strip().lower().startswith("notes:"):
                notes = line.split(":", 1)[1].strip()
            elif line.strip().lower().startswith("scale:"):
                scale_str = line.split(":", 1)[1].strip().lower()
                if scale_str in [s.value for s in Scale]:
                    scale = Scale(scale_str)
            elif line.strip().lower().startswith("tempo:"):
                try:
                    tempo = int(line.split(":", 1)[1].strip().split()[0])
                except Exception:
                    pass
        
        if not notes:
            raise ValueError("Melody agent failed to provide notes")
        if not scale:
            if "key" in constraints:
                key_parts = constraints["key"].lower().split()
                if len(key_parts) >= 2 and key_parts[1] in [s.value for s in Scale]:
                    scale = Scale(key_parts[1])
            if not scale:
                scale = Scale.MINOR if mood.lower() in ["sad", "melancholic", "dark"] else Scale.MAJOR
        if not tempo:
            tempo = 72 if style.value == "jazz" and mood.lower() in ["sad", "melancholic"] else 100
        
        return MelodyIdea(notes=notes, scale=scale, tempo=tempo)

class HarmonyAgent(MusicAgent):
    def __init__(self, llm):
        system_prompt = (
            "You are a Harmony Agent with deep music theory knowledge. "
            "Generate chord progressions with chord symbols and key information. "
            "If constraints provide a specific key, use that key."
        )
        super().__init__("Harmony Agent", llm, system_prompt)
    
    def create_harmony(self, melody: MelodyIdea, style: MusicalStyle, constraints: Optional[Dict[str, Any]] = None) -> HarmonyProgression:
        constraints_text = f"\n- Additional constraints: {constraints}" if constraints else ""
        prompt = f"""
        Create chord progressions to harmonize the following melody:
        - Melody notes: {melody.notes}
        - Scale/mode: {melody.scale.value}
        - Style: {style.value}{constraints_text}
        
        Format your response exactly as:
        CHORDS: Fm7 | Bbm7 | Cm7 ...
        KEY: F minor
        """
        response = self.generate_response(prompt)
        chords, key = None, None
        
        constraint_key = constraints.get("key") if constraints and "key" in constraints else None
        
        for line in response.split('\n'):
            if line.strip().lower().startswith("chords:"):
                chords = line.split(":", 1)[1].strip()
            elif line.strip().lower().startswith("key:"):
                key = line.split(":", 1)[1].strip()
        
        if not chords:
            raise ValueError("Harmony agent failed to provide chord progression")
        if not key:
            key = constraint_key if constraint_key else "F minor"
        
        return HarmonyProgression(chords=chords, key=key)

class RhythmAgent(MusicAgent):
    def __init__(self, llm):
        system_prompt = (
            "You are a Rhythm Agent focused on creating clear rhythmic patterns and engaging beat patterns. "
            "Generate both a pattern (using symbols like 'x' for hits and '-' for rests) and a separate beat pattern "
            "(using symbols like 'B' for bass drum, 'S' for snare, etc.) that reflect the mood."
        )
        super().__init__("Rhythm Agent", llm, system_prompt)
    
    def create_rhythm(self, style: MusicalStyle, melody: MelodyIdea, mood: str, constraints: Optional[Dict[str, Any]] = None) -> RhythmPattern:
        constraints_text = f"\n- Additional constraints: {constraints}" if constraints else ""
        prompt = f"""
        Create a rhythmic pattern with:
        - Style: {style.value}
        - Mood: {mood}
        - Tempo: {melody.tempo} BPM{constraints_text}
        
        Format your response exactly as:
        PATTERN: x - x - ... 
        TIME SIGNATURE: 4/4
        GROOVE: Description of the groove
        BEATS: B S B S ... (describe the percussion beat pattern reflecting the mood)
        """
        response = self.generate_response(prompt)
        pattern, time_signature, groove, beats = None, None, None, None
        
        for line in response.split('\n'):
            lower_line = line.strip().lower()
            if lower_line.startswith("pattern:"):
                pattern = line.split(":", 1)[1].strip()
            elif lower_line.startswith("time signature:"):
                time_signature = line.split(":", 1)[1].strip()
            elif lower_line.startswith("groove:"):
                groove = line.split(":", 1)[1].strip()
            elif lower_line.startswith("beats:"):
                beats = line.split(":", 1)[1].strip()
        
        if not pattern:
            raise ValueError("Rhythm agent failed to provide pattern")
        if not time_signature:
            time_signature = "4/4"
        if not groove:
            groove = "Moderate groove"
        if not beats:
            # Provide a default beat pattern based on mood
            beats = "B - S -" if mood.lower() in ["ecstatic", "happy", "upbeat"] else "B S - -" 
            
        return RhythmPattern(pattern=pattern, time_signature=time_signature, groove=groove, beats=beats)

class ArrangementAgent(MusicAgent):
    def __init__(self, llm):
        system_prompt = (
            "You are an Arrangement Agent specializing in orchestration. "
            "Provide instrumentation choices, overall structure, and dynamic plans. "
            "If any element (especially dynamics) is missing, supply a sensible default."
        )
        super().__init__("Arrangement Agent", llm, system_prompt)
    
    def create_arrangement(self, composition: Composition, style: MusicalStyle, constraints: Optional[Dict[str, Any]] = None) -> Arrangement:
        comp_summary = {
            "melody": composition.melody.to_dict() if composition.melody else "None",
            "harmony": composition.harmony.to_dict() if composition.harmony else "None",
            "rhythm": composition.rhythm.to_dict() if composition.rhythm else "None"
        }
        constraints_text = f"\n- Additional constraints: {constraints}" if constraints else ""
        prompt = f"""
        Create an instrumental arrangement for this composition:
        - Style: {style.value}
        - Melody: {comp_summary['melody']}
        - Harmony: {comp_summary['harmony']}
        - Rhythm: {comp_summary['rhythm']}{constraints_text}
        
        Format your response exactly as:
        INSTRUMENTATION: piano, bass, drums, etc.
        STRUCTURE: Intro - A - B - A - Outro
        DYNAMICS: Moderate dynamics; crescendo in the bridge, etc.
        """
        response = self.generate_response(prompt)
        
        instrumentation, structure, dynamics = None, None, None
        
        for line in response.split('\n'):
            lower_line = line.strip().lower()
            if lower_line.startswith("instrumentation:"):
                instruments_text = line.split(":", 1)[1].strip()
                instrumentation = [i.strip() for i in instruments_text.split(',')]
            elif lower_line.startswith("structure:"):
                structure = line.split(":", 1)[1].strip()
            elif lower_line.startswith("dynamics:"):
                dynamics = line.split(":", 1)[1].strip()
        
        if not instrumentation:
            raise ValueError("Arrangement agent failed to provide instrumentation")
        if not structure:
            raise ValueError("Arrangement agent failed to provide structure")
        if not dynamics:
            dynamics = "Moderate dynamics throughout with a slight crescendo in the bridge."
            logger.warning("Arrangement agent did not provide dynamics; using default.")
            
        return Arrangement(instrumentation=instrumentation, structure=structure, dynamics=dynamics)

class CriticAgent(MusicAgent):
    def __init__(self, llm):
        system_prompt = (
            "You are a Critic Agent with deep musical knowledge. "
            "Evaluate compositions and provide specific, constructive feedback that is not generic."
        )
        super().__init__("Critic Agent", llm, system_prompt)
    
    def evaluate_composition(self, composition: Composition) -> List[str]:
        comp_dict = composition.to_dict()
        prompt = (
            "Evaluate this musical composition and provide constructive feedback only if necessary:\n\n" +
            "COMPOSITION:\n" + json.dumps(comp_dict, indent=2) + "\n\n"
        )
        response = self.generate_response(prompt)
        feedback = []
        in_feedback_section = False
        
        for line in response.split('\n'):
            line = line.strip()
            if line.lower().startswith(("feedback:", "suggestions:", "improvements:")):
                in_feedback_section = True
                continue
            if in_feedback_section and line and not line.startswith("#"):
                cleaned_line = line
                if line[0].isdigit() and line[1:3] in ['. ', ') ']:
                    cleaned_line = line[3:].strip()
                elif line[0] in ['-', '*', '•']:
                    cleaned_line = line[1:].strip()
                if cleaned_line:
                    feedback.append(cleaned_line)
        
        if not feedback:
            feedback = [response]
        
        return feedback

# LlamaIndex Tools for Agent Communication
def create_melody_tool(melody_agent: MelodyAgent):
    def melody_tool(style: str, mood: str, constraints: str = "{}") -> str:
        try:
            style_enum = MusicalStyle(style.lower())
            constraints_dict = json.loads(constraints) if constraints else {}
            melody = melody_agent.create_melody(style_enum, mood, constraints_dict)
            return json.dumps(melody.to_dict())
        except Exception as e:
            return f"Error creating melody: {str(e)}"
    
    return FunctionTool.from_defaults(
        name="create_melody",
        fn=melody_tool,
        description="Creates a musical melody based on the specified style and mood. "
                    "Constraints (a JSON string) may include keys such as 'max_length', 'key', 'phrase_structure', 'contour', and 'harmony_emphasis'."
    )

def create_harmony_tool(harmony_agent: HarmonyAgent):
    def harmony_tool(melody_json: str, style: str, constraints: str = "{}") -> str:
        try:
            melody_dict = json.loads(melody_json)
            melody = MelodyIdea.from_dict(melody_dict)
            style_enum = MusicalStyle(style.lower())
            constraints_dict = json.loads(constraints) if constraints else {}
            harmony = harmony_agent.create_harmony(melody, style_enum, constraints_dict)
            return json.dumps(harmony.to_dict())
        except Exception as e:
            return f"Error creating harmony: {str(e)}"
    
    return FunctionTool.from_defaults(
        name="create_harmony",
        fn=harmony_tool,
        description="Creates chord progressions to harmonize a melody. "
                    "Constraints (a JSON string) may include keys such as 'key' and other style notes."
    )

def create_rhythm_tool(rhythm_agent: RhythmAgent):
    def rhythm_tool(style: str, melody_json: str, mood: str, constraints: str = "{}") -> str:
        try:
            melody_dict = json.loads(melody_json)
            melody = MelodyIdea.from_dict(melody_dict)
            style_enum = MusicalStyle(style.lower())
            constraints_dict = json.loads(constraints) if constraints else {}
            rhythm = rhythm_agent.create_rhythm(style_enum, melody, mood, constraints_dict)
            return json.dumps(rhythm.to_dict())
        except Exception as e:
            return f"Error creating rhythm: {str(e)}"
    
    return FunctionTool.from_defaults(
        name="create_rhythm",
        fn=rhythm_tool,
        description="Creates rhythmic patterns (including a beat pattern) appropriate for a melody and style. "
                    "Constraints (a JSON string) may include additional notes on the rhythmic feel."
    )

def create_arrangement_tool(arrangement_agent: ArrangementAgent):
    def arrangement_tool(composition_json: str, style: str, constraints: str = "{}") -> str:
        try:
            comp_dict = json.loads(composition_json)
            composition = Composition.from_dict(comp_dict)
            style_enum = MusicalStyle(style.lower())
            constraints_dict = json.loads(constraints) if constraints else {}
            arrangement = arrangement_agent.create_arrangement(composition, style_enum, constraints_dict)
            return json.dumps(arrangement.to_dict())
        except Exception as e:
            return f"Error creating arrangement: {str(e)}"
    
    return FunctionTool.from_defaults(
        name="create_arrangement",
        fn=arrangement_tool,
        description="Creates an instrumental arrangement for a composition. "
                    "Constraints (a JSON string) may include further instructions on instrumentation, structure, and dynamics."
    )

def create_critic_tool(critic_agent: CriticAgent):
    def critic_tool(composition_json: str) -> str:
        try:
            comp_dict = json.loads(composition_json)
            composition = Composition.from_dict(comp_dict)
            feedback = critic_agent.evaluate_composition(composition)
            return json.dumps({"feedback": feedback})
        except Exception as e:
            return f"Error evaluating composition: {str(e)}"
    
    return FunctionTool.from_defaults(
        name="evaluate_composition",
        fn=critic_tool,
        description="Evaluates a musical composition and provides constructive feedback."
    )

def create_render_score_tool():
    def render_score_tool(composition_json: str, output_path: str = "score.xml") -> str:
        try:
            comp_dict = json.loads(composition_json)
            composition = Composition.from_dict(comp_dict)
            
            score = music21.stream.Score()
            metadata = music21.metadata.Metadata()
            metadata.title = composition.title
            score.metadata = metadata
            
            part = music21.stream.Part()
            if composition.melody:
                note_strings = composition.melody.notes.split()
                for note_str in note_strings:
                    note = music21.note.Note(note_str)
                    part.append(note)
            score.append(part)
            
            directory = os.path.dirname(output_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            score.write('musicxml', fp=output_path)
            
            return json.dumps({"score": f"MusicXML score generated at {output_path}"})
            
        except Exception as e:
            return f"Error rendering score: {str(e)}"
    
    return FunctionTool.from_defaults(
        name="render_score",
        fn=render_score_tool,
        description="Renders the composition to a MusicXML score."
    )

# New tool: Generate audio (WAV or MP3) directly from the composition.
# This tool first renders a MIDI file from the Music21 score, then uses FluidSynth (and optionally ffmpeg) for conversion.
def create_generate_audio_tool():
    def generate_audio_tool(composition_json: str, output_path: str = "output.wav", sf2_path: str = "FluidR3_GM.sf2") -> str:
        try:
            comp_dict = json.loads(composition_json)
            composition = Composition.from_dict(comp_dict)
            
            # Create a Music21 score (same as before)
            score = music21.stream.Score()
            metadata = music21.metadata.Metadata()
            metadata.title = composition.title
            score.metadata = metadata
            
            if composition.melody and composition.melody.tempo:
                mm = music21.tempo.MetronomeMark(number=composition.melody.tempo)
                score.append(mm)
            
            instruments = ["piano"]
            if composition.arrangement and composition.arrangement.instrumentation:
                instruments = composition.arrangement.instrumentation
            
            if composition.melody:
                melody_part = music21.stream.Part()
                melody_inst = music21.instrument.fromString(instruments[0])
                melody_part.append(melody_inst)
                
                if composition.rhythm and composition.rhythm.time_signature:
                    ts_parts = composition.rhythm.time_signature.split('/')
                    if len(ts_parts) == 2:
                        try:
                            numerator = int(ts_parts[0])
                            denominator = int(ts_parts[1])
                            time_sig = music21.meter.TimeSignature(f"{numerator}/{denominator}")
                            melody_part.append(time_sig)
                        except ValueError:
                            pass
                
                note_strings = composition.melody.notes.split()
                for note_str in note_strings:
                    try:
                        if ',' in note_str:
                            pitch_str, duration_str = note_str.split(',')
                            note = music21.note.Note(pitch_str)
                            if duration_str == 'w':
                                note.quarterLength = 4.0
                            elif duration_str == 'h':
                                note.quarterLength = 2.0
                            elif duration_str == 'q':
                                note.quarterLength = 1.0
                            elif duration_str == 'e':
                                note.quarterLength = 0.5
                            elif duration_str == 's':
                                note.quarterLength = 0.25
                        else:
                            note = music21.note.Note(note_str)
                        melody_part.append(note)
                    except Exception as e:
                        logger.warning(f"Error parsing note {note_str}: {e}")
                        continue
                
                score.append(melody_part)
            
            if composition.harmony and len(instruments) > 1:
                harmony_part = music21.stream.Part()
                harmony_inst = music21.instrument.fromString(instruments[1])
                harmony_part.append(harmony_inst)
                
                chord_strings = composition.harmony.chords.split('|')
                for chord_str in chord_strings:
                    chord_str = chord_str.strip()
                    if not chord_str:
                        continue
                    try:
                        chord = music21.harmony.ChordSymbol(chord_str)
                        harmony_part.append(chord)
                    except Exception as e:
                        logger.warning(f"Error parsing chord {chord_str}: {e}")
                        continue
                
                score.append(harmony_part)
            
            if composition.rhythm and composition.rhythm.pattern and len(instruments) > 2:
                rhythm_part = music21.stream.Part()
                rhythm_inst = music21.instrument.fromString("Percussion")
                rhythm_part.append(rhythm_inst)
                
                pattern = composition.rhythm.pattern
                for symbol in pattern:
                    if symbol == 'x':
                        perc_note = music21.note.Note('C2')
                        rhythm_part.append(perc_note)
                    elif symbol == '-':
                        rest = music21.note.Rest()
                        rhythm_part.append(rest)
                
                score.append(rhythm_part)
            
            # Write an intermediate MIDI file
            temp_midi = "temp_output.mid"
            directory = os.path.dirname(temp_midi)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            score.write('midi', fp=temp_midi)
            
            # Use FluidSynth to convert MIDI to WAV.
            # This command assumes FluidSynth and the specified SoundFont (sf2) are correctly installed.
            command = ["fluidsynth", "-ni", sf2_path, temp_midi, "-F", output_path]
            subprocess.run(command, check=True)
            
            # If the desired output is MP3, use ffmpeg to convert
            if output_path.lower().endswith(".mp3"):
                wav_temp = "temp_output.wav"
                # First generate a WAV file
                command = ["fluidsynth", "-ni", sf2_path, temp_midi, "-F", wav_temp]
                subprocess.run(command, check=True)
                # Convert WAV to MP3 using ffmpeg
                command = ["ffmpeg", "-y", "-i", wav_temp, output_path]
                subprocess.run(command, check=True)
                os.remove(wav_temp)
            
            os.remove(temp_midi)
            composition.audio_path = output_path
            
            return json.dumps({
                "audio_path": output_path,
                "message": f"Audio file generated at {output_path}"
            })
            
        except Exception as e:
            return f"Error generating audio: {str(e)}"
    
    return FunctionTool.from_defaults(
        name="generate_audio",
        fn=generate_audio_tool,
        description="Generates an audio file (WAV or MP3) from the composition using FluidSynth and ffmpeg. "
                    "Requires a valid SoundFont file and external command-line tools."
    )

# Orchestrator class using LlamaIndex
class CompositionOrchestrator:
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.setup_agents()
        self.setup_tools()
        self.setup_orchestrator()
    
    def setup_agents(self):
        gemini_llm = Gemini(
            api_key=self.api_keys.get("gemini"),
            model="models/gemini-2.0-flash-thinking-exp-01-21"
        )
        self.melody_agent = MelodyAgent(gemini_llm)
        self.harmony_agent = HarmonyAgent(gemini_llm)
        self.rhythm_agent = RhythmAgent(gemini_llm)
        self.arrangement_agent = ArrangementAgent(gemini_llm)
        self.critic_agent = CriticAgent(gemini_llm)
    
    def setup_tools(self):
        self.tools = [
            create_melody_tool(self.melody_agent),
            create_harmony_tool(self.harmony_agent),
            create_rhythm_tool(self.rhythm_agent),
            create_arrangement_tool(self.arrangement_agent),
            create_critic_tool(self.critic_agent),
            create_render_score_tool(),
            create_generate_audio_tool()
        ]
    
    def setup_orchestrator(self):
        gemini_llm = Gemini(
            api_key=self.api_keys.get("gemini"),
            model="models/gemini-2.0-flash-thinking-exp-01-21"
        )
        
        agent_prompt = (
            "You are a Music Composition Orchestrator that coordinates specialized music agents.\n"
            "Workflow:\n"
            "1. Create a melody\n"
            "2. Create harmony\n"
            "3. Create rhythm (including a beat pattern)\n"
            "4. Create an arrangement\n"
            "5. Evaluate the composition\n"
            "6. Improve based on feedback\n"
            "7. Render the final score\n"
            "8. Generate an audio file (WAV or MP3) from the composition"
        )
        
        self.orchestrator = ReActAgent.from_tools(
            self.tools,
            llm=gemini_llm,
            system_prompt=agent_prompt,
            verbose=True
        )
    
    def compose_music(self, style: str, mood: str, title: str, constraints: Dict[str, Any] = None) -> (Composition, Any):
        constraints_str = json.dumps(constraints) if constraints else "{}"
        
        prompt = f"""
        Please compose a complete piece of music with the following specifications:
        - Title: {title}
        - Style: {style}
        - Mood: {mood}
        - Additional constraints: {constraints_str}
        
        Create a composition by coordinating all the specialized music agents in sequence.
        Start with melody, then harmony, rhythm, arrangement, and finally get feedback from the critic.
        After 1 improvement iteration, render the score and generate an audio file from the composition.
        """
        time.sleep(5)
        response = self.orchestrator.query(prompt)
        
        composition = Composition(title=title)
        text = response.response.strip()
        try:
            comp_dict = json.loads(text)
            composition = Composition.from_dict(comp_dict)
        except Exception:
            return composition.audio_path

        
        return composition.audio_path

# Example usage of the system
def example_usage():
    api_keys = {
        "gemini": <GEMINI_API_KEY>,
        "mistral": <MISTRAL_API_KEY>
    }
    
    orchestrator = CompositionOrchestrator(api_keys)
    
    composition, response = orchestrator.compose_music(
        style="classical",
        mood="happy",
        title="Spring"
        # You could also pass constraints here, e.g.: {"max_length": 10, "key": "C major"}
    )
    
    print("Composition process:")
    print(response.response)
    print("\nFinal composition:")
    print(json.dumps(composition.to_dict(), indent=2))
    
    if composition.audio_path and os.path.exists(composition.audio_path):
        print(f"\nAudio file generated: {composition.audio_path}")
    else:
        print("\nNo audio file was generated.")

if __name__ == "__main__":
    example_usage()
