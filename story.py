import streamlit as st
from typing import TypedDict,Literal
from langgraph.graph import StateGraph,END
from groq import Groq
import os
import json
from dotenv import load_dotenv
load_dotenv()  # reads .env file
api_key = os.getenv("api_key")




# State definition for our comic creator
class ComicState(TypedDict):
    hero_name: str
    hero_power: str
    villain_name: str
    villain_power: str
    setting: str
    tone: str
    comic_title: str
    panel_1: str  # Introduction
    panel_2: str  # Rising action
    panel_3: str  # Climax
    panel_4: str  # Resolution
    comic_tagline: str
    current_step: str


# Initialize Groq client
def get_groq_client():
    # Add your Groq API key here

    return Groq(api_key=api_key)


# Node functions for LangGraph
def generate_title(state: ComicState) -> ComicState:
    """Generate an epic comic book title"""
    client=get_groq_client()

    prompt=f"""Create an EPIC comic book title for this story:
    Hero: {state['hero_name']} with power of {state['hero_power']}
    Villain: {state['villain_name']} with power of {state['villain_power']}
    Setting: {state['setting']}
    Tone: {state['tone']}

    Return ONLY the title, make it exciting and dramatic! Max 6 words."""

    response=client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [{"role":"user","content":prompt}],
        temperature = 1.2,
        max_tokens = 50
    )

    state['comic_title']=response.choices[0].message.content.strip().strip('"')
    state['current_step']="panel_1"
    return state


def create_panel_1(state: ComicState) -> ComicState:
    """Create introduction panel"""
    client=get_groq_client()

    prompt=f"""You're writing Panel 1 of a {state['tone']} comic book titled "{state['comic_title']}".

    INTRODUCE:
    - Hero: {state['hero_name']} (Power: {state['hero_power']})
    - Setting: {state['setting']}

    Write 2-3 exciting sentences that show our hero in their world. Make it visual and engaging for kids!
    Use action words and vivid descriptions."""

    response=client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [{"role":"user","content":prompt}],
        temperature = 1.1,
        max_tokens = 150
    )

    state['panel_1']=response.choices[0].message.content.strip()
    state['current_step']="panel_2"
    return state


def create_panel_2(state: ComicState) -> ComicState:
    """Create rising action panel"""
    client=get_groq_client()

    prompt=f"""Panel 2 of "{state['comic_title']}" - The villain appears!

    Previous: {state['panel_1']}

    NOW INTRODUCE:
    - Villain: {state['villain_name']} (Power: {state['villain_power']})
    - The problem/conflict they create

    Tone: {state['tone']}

    Write 2-3 sentences showing the villain's dramatic entrance and evil plan. Build tension!"""

    response=client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [{"role":"user","content":prompt}],
        temperature = 1.1,
        max_tokens = 150
    )

    state['panel_2']=response.choices[0].message.content.strip()
    state['current_step']="panel_3"
    return state


def create_panel_3(state: ComicState) -> ComicState:
    """Create climax panel"""
    client=get_groq_client()

    prompt=f"""Panel 3 of "{state['comic_title']}" - THE EPIC SHOWDOWN!

    Story so far:
    Panel 1: {state['panel_1']}
    Panel 2: {state['panel_2']}

    Hero: {state['hero_name']} (Power: {state['hero_power']})
    Villain: {state['villain_name']} (Power: {state['villain_power']})
    Tone: {state['tone']}

    Write 3-4 sentences of the MOST EXCITING BATTLE! Show powers clashing, creative moves, and high stakes!"""

    response=client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [{"role":"user","content":prompt}],
        temperature = 1.2,
        max_tokens = 200
    )

    state['panel_3']=response.choices[0].message.content.strip()
    state['current_step']="panel_4"
    return state


def create_panel_4(state: ComicState) -> ComicState:
    """Create resolution panel"""
    client=get_groq_client()

    prompt=f"""Panel 4 of "{state['comic_title']}" - VICTORY AND RESOLUTION!

    The battle: {state['panel_3']}

    Hero: {state['hero_name']} (Power: {state['hero_power']})
    Tone: {state['tone']}

    Write 2-3 sentences showing:
    - How {state['hero_name']} wins
    - What happens after
    - A feel-good ending (but hint at future adventures!)"""

    response=client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [{"role":"user","content":prompt}],
        temperature = 1.0,
        max_tokens = 150
    )

    state['panel_4']=response.choices[0].message.content.strip()
    state['current_step']="tagline"
    return state


def create_tagline(state: ComicState) -> ComicState:
    """Create an epic tagline"""
    client=get_groq_client()

    prompt=f"""Create a SHORT, PUNCHY tagline for this comic: "{state['comic_title']}"

    Hero: {state['hero_name']} with {state['hero_power']}

    Make it inspirational, memorable, and exciting! Maximum 8 words. Think Marvel/DC style!"""

    response=client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [{"role":"user","content":prompt}],
        temperature = 1.3,
        max_tokens = 30
    )

    state['comic_tagline']=response.choices[0].message.content.strip().strip('"')
    state['current_step']="end"
    return state


# Build the LangGraph workflow
def create_comic_graph():
    workflow=StateGraph(ComicState)

    workflow.add_node("title",generate_title)
    workflow.add_node("panel_1",create_panel_1)
    workflow.add_node("panel_2",create_panel_2)
    workflow.add_node("panel_3",create_panel_3)
    workflow.add_node("panel_4",create_panel_4)
    workflow.add_node("tagline",create_tagline)

    workflow.set_entry_point("title")
    workflow.add_edge("title","panel_1")
    workflow.add_edge("panel_1","panel_2")
    workflow.add_edge("panel_2","panel_3")
    workflow.add_edge("panel_3","panel_4")
    workflow.add_edge("panel_4","tagline")
    workflow.add_edge("tagline",END)

    return workflow.compile()


# Streamlit App
def main():
    st.set_page_config(page_title = "🦸 Comic Creator",page_icon = "💥",layout = "wide")

    # Custom CSS for comic book style
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bangers&display=swap');

        .comic-title {
            font-family: 'Bangers', cursive;
            font-size: 60px !important;
            color: #FF0000;
            text-align: center;
            text-shadow: 4px 4px 0px #000000;
            -webkit-text-stroke: 2px black;
            margin: 20px 0;
        }

        .panel {
            background: linear-gradient(135deg, #FFF9C4 0%, #FFEB3B 100%);
            border: 5px solid #000000;
            border-radius: 15px;
            padding: 25px;
            margin: 15px 0;
            box-shadow: 8px 8px 0px rgba(0,0,0,0.3);
            position: relative;
        }

        .panel:before {
            content: '';
            position: absolute;
            top: -5px;
            left: -5px;
            right: -5px;
            bottom: -5px;
            background: #000;
            border-radius: 15px;
            z-index: -1;
        }

        .panel-number {
            font-family: 'Bangers', cursive;
            font-size: 32px;
            color: #FF0000;
            -webkit-text-stroke: 1px black;
        }

        .tagline {
            font-family: 'Bangers', cursive;
            font-size: 36px;
            color: #2196F3;
            text-align: center;
            -webkit-text-stroke: 1px black;
            margin: 30px 0;
        }

        .hero-stat {
            background: #4CAF50;
            color: white;
            padding: 10px;
            border-radius: 10px;
            border: 3px solid black;
            font-weight: bold;
        }

        .villain-stat {
            background: #9C27B0;
            color: white;
            padding: 10px;
            border-radius: 10px;
            border: 3px solid black;
            font-weight: bold;
        }
        </style>
    """,unsafe_allow_html = True)

    st.markdown('<p class="comic-title">💥 Leroy AI COMIC BOOK CREATOR 💥</p>',unsafe_allow_html = True)
    st.markdown("### 🎨 Build Your Own AI-Powered Comic Book! 🦸")

    with st.sidebar:
        st.header("🎭 Design Your Comic")

        hero_name=st.text_input("🦸 Hero Name","Thunder Kid")
        hero_power=st.selectbox("⚡ Hero Power",[
            "Super Speed","Lightning Control","Invisibility",
            "Super Strength","Telepathy","Time Control",
            "Shape Shifting","Fire Power","Ice Power"
        ])

        st.markdown("---")

        villain_name=st.text_input("😈 Villain Name","Dr. Chaos")
        villain_power=st.selectbox("💀 Villain Power",[
            "Mind Control","Dark Magic","Robot Army",
            "Weather Control","Shadow Powers","Gravity Control",
            "Energy Drain","Toxic Gas","Size Manipulation"
        ])

        st.markdown("---")

        setting=st.selectbox("🌍 Setting",[
            "Futuristic Mega City","Mystical Forest Kingdom",
            "Underwater Atlantis","Space Station Omega",
            "Volcanic Mountain Range","Frozen Arctic Base",
            "Desert Pyramid Complex","Floating Sky Islands"
        ])

        tone=st.select_slider("🎬 Story Tone",[
            "Funny & Silly","Exciting Adventure",
            "Epic & Dramatic","Dark & Mysterious"
        ])

        st.markdown("---")
        create_button=st.button("🚀 CREATE MY COMIC!",type = "primary",use_container_width = True)

    # Main content
    if create_button:
        # Initialize state
        initial_state=ComicState(
            hero_name = hero_name,
            hero_power = hero_power,
            villain_name = villain_name,
            villain_power = villain_power,
            setting = setting,
            tone = tone,
            comic_title = "",
            panel_1 = "",
            panel_2 = "",
            panel_3 = "",
            panel_4 = "",
            comic_tagline = "",
            current_step = "title"
        )

        # Progress tracking
        progress_container=st.container()
        with progress_container:
            st.markdown("### ⚡ AI is Creating Your Comic...")
            progress_bar=st.progress(0)
            status=st.empty()

            steps=["title","panel_1","panel_2","panel_3","panel_4","tagline"]
            step_names=["Generating Title","Creating Panel 1","Creating Panel 2",
                        "Creating Panel 3","Creating Panel 4","Adding Tagline"]

        # Create graph and run
        comic_graph=create_comic_graph()

        # Execute with progress updates
        result=None
        for i,(step,step_name) in enumerate(zip(steps,step_names)):
            status.markdown(f"**{step_name}...** 🎨")
            progress_bar.progress((i + 1) / len(steps))

            if i == 0:
                result=comic_graph.invoke(initial_state)

        progress_container.empty()

        # Display the comic
        st.balloons()

        st.markdown(f'<p class="comic-title">{result["comic_title"]}</p>',unsafe_allow_html = True)

        # Character cards
        col1,col2=st.columns(2)
        with col1:
            st.markdown(f'<div class="hero-stat">🦸 {hero_name}<br>⚡ {hero_power}</div>',unsafe_allow_html = True)
        with col2:
            st.markdown(f'<div class="villain-stat">😈 {villain_name}<br>💀 {villain_power}</div>',
                        unsafe_allow_html = True)

        st.markdown("---")

        # Panels
        panels=[
            ("Panel 1: The Beginning",result['panel_1']),
            ("Panel 2: The Threat",result['panel_2']),
            ("Panel 3: The Battle!",result['panel_3']),
            ("Panel 4: Victory!",result['panel_4'])
        ]

        for i,(title,content) in enumerate(panels,1):
            st.markdown(f"""
                <div class="panel">
                    <div class="panel-number">{title}</div>
                    <p style="font-size: 18px; line-height: 1.6; margin-top: 10px; color: #000000;">{content}</p>
                </div>
            """,unsafe_allow_html = True)

        st.markdown(f'<p class="tagline">"{result["comic_tagline"]}"</p>',unsafe_allow_html = True)

        # Download options
        st.markdown("---")
        col1,col2,col3=st.columns(3)

        with col1:
            # Text format
            comic_text=f"""{result['comic_title'].upper()}

🦸 HERO: {hero_name} - {hero_power}
😈 VILLAIN: {villain_name} - {villain_power}
🌍 SETTING: {setting}

PANEL 1: THE BEGINNING
{result['panel_1']}

PANEL 2: THE THREAT
{result['panel_2']}

PANEL 3: THE BATTLE!
{result['panel_3']}

PANEL 4: VICTORY!
{result['panel_4']}

"{result['comic_tagline']}"

Created with AI Comic Book Creator
"""
            st.download_button(
                "📥 Download as Text",
                comic_text,
                file_name = f"{result['comic_title'].replace(' ','_')}.txt",
                mime = "text/plain",
                use_container_width = True
            )

        with col2:
            # JSON format for programmers
            comic_json=json.dumps(result,indent = 2)
            st.download_button(
                "💾 Download as JSON",
                comic_json,
                file_name = f"{result['comic_title'].replace(' ','_')}.json",
                mime = "application/json",
                use_container_width = True
            )

        with col3:
            if st.button("🔄 Create Another!",use_container_width = True):
                st.rerun()

        # Achievement
        st.success(
            "🏆 ACHIEVEMENT UNLOCKED: Comic Book Creator! You just used AI and graph technology to build something amazing!")

    else:
        st.info("👈 Design your characters and click 'CREATE MY COMIC!' to begin!")
        st.markdown("""
        ### 🎯 What You'll Create:
        - ✨ **AI-Generated Title** - Groq creates an epic name!
        - 📖 **4 Comic Panels** - A complete story arc
        - 💥 **Epic Tagline** - Your comic's catchphrase
        - 🎨 **Professional Format** - Like a real comic book!

        ### 🤖 Powered By:
        - **LangGraph** - AI workflow that builds your story step-by-step
        - **Groq** - Lightning-fast AI that generates creative content
        - **Your Imagination** - The most important ingredient!
        """)


if __name__ == "__main__":
    main()