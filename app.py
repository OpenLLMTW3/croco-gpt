import streamlit as st
import pandas as pd
import PyPDF2
import openai
import os
from dotenv import load_dotenv
from PIL import Image
import io
import math
import ast
import operator
import json

load_dotenv()
openai.api_key = st.secrets.get("OPENAI_API_KEY")

MAX_TOKENS = 100000

def main():
    st.set_page_config(page_title="Training GPT", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stTextInput > div > div > input {
        background-color: #2D2D2D;
        color: #FFFFFF;
    }
    .send-button {
        background-color: white !important;
        color: black !important;
    }
    .logo-container {
        position: fixed;
        top: 10px;
        left: 10px;
        z-index: 1000;
    }
    .title {
        text-align: center;
        padding-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Add Lacoste logo placeholder
    st.markdown("""
    <div class="logo-container">
        <img src="https://ibb.co/wck8ZHR" alt="Lacoste Logo" style="width:100px;"/>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='title'>Training GPT</h1>", unsafe_allow_html=True)

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'file_contents' not in st.session_state:
        st.session_state.file_contents = None
    if 'system_prompt' not in st.session_state:
        st.session_state.system_prompt = """Vous êtes un assistant AI capable d'analyser le contenu des fichiers, y compris les images, et de répondre aux questions. 
        Utilisez les informations du fichier chargé pour répondre aux questions de l'utilisateur. 
        Vous avez également des capacités avancées en calcul pour l'analyse comptable."""
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7

    # Sidebar
    with st.sidebar:
        st.title("Options")
        uploaded_file = st.file_uploader("Choisissez un fichier (CSV, PDF)", type=["csv", "pdf"])
        if uploaded_file is not None:
            st.session_state.file_contents = process_file(uploaded_file)
            if st.session_state.file_contents:
                st.success("Fichier chargé avec succès!")
            else:
                st.error("Erreur lors du chargement du fichier.")
        
        st.session_state.system_prompt = st.text_area("Personnalisez le prompt système:", value=st.session_state.system_prompt, height=150)
        
        st.session_state.temperature = st.slider("Température de l'IA (créativité):", min_value=0.0, max_value=1.0, value=st.session_state.temperature, step=0.1)
        
        if st.button("Réinitialiser la conversation"):
            st.session_state.messages = []
            st.session_state.file_contents = None
            st.experimental_rerun()

    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Votre message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in get_ai_response(prompt):
                full_response = response
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

def process_file(uploaded_file):
    try:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            return truncate_text(df.to_string())
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return truncate_text(text)
        else:
            st.error("Type de fichier non supporté")
            return None
    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier: {str(e)}")
        return None

def truncate_text(text):
    if len(text) > MAX_TOKENS * 4:
        return text[:MAX_TOKENS * 4] + "... (contenu tronqué)"
    return text

def safe_eval(expr):
    """Évalue de manière sécurisée une expression mathématique."""
    operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                 ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}
    
    def eval_expr(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
        elif isinstance(node, ast.UnaryOp):
            return operators[type(node.op)](eval_expr(node.operand))
        else:
            raise TypeError(node)
    
    return eval_expr(ast.parse(expr, mode='eval').body)

def perform_calculation(expression):
    try:
        # Remplacer les virgules par des points pour les décimaux
        expression = expression.replace(',', '.')
        
        # Évaluation sécurisée de l'expression
        result = safe_eval(expression)
        
        # Arrondir le résultat à 2 décimales
        return round(result, 2)
    except Exception as e:
        return f"Erreur de calcul : {str(e)}"

def get_ai_response(user_input):
    messages = [
        {"role": "system", "content": st.session_state.system_prompt}
    ]
    
    if st.session_state.file_contents:
        messages.append({"role": "user", "content": f"Voici le contenu du fichier à analyser:\n\n{st.session_state.file_contents}\n\nVeuillez analyser ce contenu et répondre aux questions suivantes."})
    
    for message in st.session_state.messages:
        messages.append(message)
    
    messages.append({"role": "user", "content": user_input})
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=st.session_state.temperature,
            stream=True,
            functions=[{
                "name": "perform_calculation",
                "description": "Effectue des calculs mathématiques",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "L'expression mathématique à évaluer"
                        }
                    },
                    "required": ["expression"]
                }
            }],
            function_call="auto"
        )
        
        full_response = ""
        function_name = None
        function_args = ""

        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.function_call:
                if delta.function_call.name:
                    function_name = delta.function_call.name
                if delta.function_call.arguments:
                    function_args += delta.function_call.arguments
            elif delta.content:
                full_response += delta.content
                yield full_response

        if function_name == "perform_calculation":
            try:
                args = json.loads(function_args)
                result = perform_calculation(args["expression"])
                yield f"{full_response}\nLe résultat du calcul est : {result}"
            except json.JSONDecodeError:
                yield f"{full_response}\nErreur : Impossible de décoder les arguments de la fonction."
            except KeyError:
                yield f"{full_response}\nErreur : L'expression à calculer est manquante."
        elif not full_response:
            yield "Désolé, je n'ai pas pu générer une réponse. Pouvez-vous reformuler votre question ?"

    except Exception as e:
        yield f"Erreur lors de la communication avec l'IA: {str(e)}"

if __name__ == "__main__":
    main()
