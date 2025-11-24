# ======================================================
# 🌿 DAILY WELLNESS VOICE COMPANION — SANJAY EDITION
# 🤖 Personalized, Clean & Friendly Developer Style
# 🧭 Smooth Flow • Clear Comments • Indian Vibes
# ======================================================

import logging
import json
import os
from datetime import datetime
from typing import Annotated, List
from dataclasses import dataclass, field, asdict

print("\n" + "🌿" * 50)
print("🚀 SANJAY'S WELLNESS AGENT INITIALIZING…")
print("💚 Your Daily Companion for Mood, Energy & Goals")
print("🌿" * 50 + "\n")

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# 🧠 STATE MANAGEMENT
# ======================================================

@dataclass
class CheckInState:
    mood: str | None = None
    energy: str | None = None
    objectives: list[str] = field(default_factory=list)
    advice_given: str | None = None

    def is_complete(self) -> bool:
        return all([
            self.mood is not None,
            self.energy is not None,
            len(self.objectives) > 0,
        ])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Userdata:
    current_checkin: CheckInState
    history_summary: str
    session_start: datetime = field(default_factory=datetime.now)

# ======================================================
# 💾 JSON PERSISTENCE
# ======================================================

WELLNESS_LOG_FILE = "wellness_log.json"

def get_log_path():
    base_dir = os.path.dirname(__file__)
    backend_dir = os.path.abspath(os.path.join(base_dir, ".."))
    return os.path.join(backend_dir, WELLNESS_LOG_FILE)

def load_history() -> list:
    path = get_log_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except:
        return []

def save_checkin_entry(entry: CheckInState) -> None:
    path = get_log_path()
    history = load_history()

    record = {
        "timestamp": datetime.now().isoformat(),
        "mood": entry.mood,
        "energy": entry.energy,
        "objectives": entry.objectives,
        "summary": entry.advice_given,
    }

    history.append(record)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Saved check-in → {path}")

# ======================================================
# 🛠️ WELLNESS TOOLS
# ======================================================

@function_tool
async def record_mood_and_energy(
    ctx: RunContext[Userdata],
    mood: Annotated[str, Field(description="User mood: happy, stressed…")],
    energy: Annotated[str, Field(description="User energy: high, low…")],
) -> str:
    ctx.userdata.current_checkin.mood = mood
    ctx.userdata.current_checkin.energy = energy
    print(f"📊 Mood={mood} | Energy={energy}")
    return f"Got it — you're feeling {mood} with {energy} energy."


@function_tool
async def record_objectives(
    ctx: RunContext[Userdata],
    objectives: Annotated[List[str], Field(description="1–3 goals for the day")],
) -> str:
    ctx.userdata.current_checkin.objectives = objectives
    print("🎯 Objectives:", objectives)
    return "I've noted down your goals for today."


@function_tool
async def complete_checkin(
    ctx: RunContext[Userdata],
    final_advice_summary: Annotated[str, Field(description="1-sentence recap")],
) -> str:
    state = ctx.userdata.current_checkin
    state.advice_given = final_advice_summary

    if not state.is_complete():
        return "I still need your mood, energy, and at least one goal before finishing."

    save_checkin_entry(state)

    recap = f"""
Here’s your daily recap:
• Mood: {state.mood}
• Energy: {state.energy}
• Goals: {', '.join(state.objectives)}

Reminder: {final_advice_summary}
Your check-in has been saved. Have a great day, Sanjay! 🌿
"""
    return recap

# ======================================================
# 🤖 AGENT BEHAVIOR
# ======================================================

class WellnessAgent(Agent):
    def __init__(self, history_context: str):
        super().__init__(
            instructions=f"""
You are Sanjay's Daily Wellness Companion — warm, friendly, grounded.

🧠 PREVIOUS SESSION CONTEXT:
{history_context}

🎯 SESSION GOALS:
1. Ask how Sanjay is feeling (mood + energy).
2. Ask for 1–3 simple goals for the day.
3. Give supportive NON-medical suggestions.
4. At the end, call complete_checkin.

⚠️ SAFETY:
- You are *not* a therapist or doctor.
- No diagnoses or medical advice.
- If user expresses crisis, gently suggest professional help.
""",
            tools=[record_mood_and_energy, record_objectives, complete_checkin],
        )

# ======================================================
# 🎬 ENTRYPOINT
# ======================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    print("\n🌿 Starting Sanjay's Wellness Session…")

    history = load_history()
    history_summary = "No previous sessions yet."

    if history:
        last = history[-1]
        history_summary = (
            f"Last check-in on {last['timestamp']}. "
            f"Mood was {last['mood']}, energy {last['energy']}. "
            f"Goals: {', '.join(last['objectives'])}."
        )
        print("📜 Loaded history.")

    userdata = Userdata(
        current_checkin=CheckInState(),
        history_summary=history_summary,
    )

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-natalie",
            style="Promo",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    await session.start(
        agent=WellnessAgent(history_context=history_summary),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
