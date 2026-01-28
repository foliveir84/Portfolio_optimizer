import streamlit as st

def init_session_state():
    """Inicializa variáveis globais que devem persistir entre páginas."""
    defaults = {
        "custo_hora": 12.0,
    }
    
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def apply_custom_style():
    """Aplica o design System 'PharmaTouch Glass'."""
    st.markdown("""
    <style>
        /* Fundo e Estrutura Geral */
        .stApp {
            background: linear-gradient(135deg, #0e0b16 0%, #1a1625 100%);
        }
        
        /* Headers com Gradiente Suave */
        h1, h2, h3 {
            background: -webkit-linear-gradient(0deg, #00e5ff, #d900ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700 !important;
            letter-spacing: -0.5px;
        }
        
        /* Glassmorphism Cards (Metric Containers) */
        div[data-testid="stMetric"], .metric-container {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 15px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }
        
        /* Inputs Modernos */
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"], .stNumberInput input {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: #f0f2f6 !important; /* Texto visível */
            border-radius: 8px !important;
        }
        .stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
            border-color: #d900ff !important;
            box-shadow: 0 0 0 1px #d900ff !important;
        }
        
        /* Placeholder Visível */
        .stTextInput input::placeholder, .stTextArea textarea::placeholder {
            color: rgba(255, 255, 255, 0.4) !important;
            opacity: 1 !important;
        }
        
        /* Botões */
        .stButton button {
            border-radius: 8px;
            font-weight: 600;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            background: linear-gradient(90deg, #d900ff, #00e5ff);
            color: white;
            border: none;
        }

        /* Sidebar Background */
        section[data-testid="stSidebar"] {
            background-color: #1A1625;
        }
        
        /* DataFrame Styling Overrides */
        div[data-testid="stDataFrame"] {
            background-color: rgba(255, 255, 255, 0.02);
            border-radius: 10px;
            padding: 10px;
        }
    </style>
    """, unsafe_allow_html=True)