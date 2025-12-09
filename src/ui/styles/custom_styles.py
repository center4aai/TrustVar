# src/ui/styles/custom_styles.py
import streamlit as st


def apply_custom_styles():
    """Применить кастомные стили к приложению"""

    st.markdown(
        """
<style>
    /* ========== ОСНОВНЫЕ СТИЛИ ========== */
    .main {
        padding-top: 2rem;
    }
    
    /* Скрываем sidebar полностью */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* ========== ЗАГОЛОВОК ========== */
    .main-header {
        text-align: center;
        margin: 0 0 2rem 0;   
        padding: 1rem 0;  
    }
    
    .main-title {
        font-size: 4.5rem;
        font-weight: 800;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: #a78bfa;
        font-weight: 400;
        opacity: 0.9;
    }
    
    /* ========== КАРТОЧКИ-КНОПКИ ========== */
    .stButton > button {
        background: linear-gradient(145deg, #1e1e1e, #2a2a2a);
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-radius: 16px;
        padding: 2rem 1rem;
        height: 200px;
        width: 100%;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        color: white;
        font-weight: 500;
        text-align: center;
        white-space: pre-wrap;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(145deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        opacity: 0;
        transition: opacity 0.3s ease;
        border-radius: 16px;
    }
    
    .stButton > button:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
        border-color: #667eea;
        background: linear-gradient(145deg, #2a2a2a, #353535);
    }
    
    .stButton > button:hover::before {
        opacity: 1;
    }
    
    .stButton > button:active {
        transform: translateY(-4px) scale(1.01);
    }
    
    /* Анимация пульсации */
    @keyframes pulse {
        0%, 100% {
            box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
        }
        50% {
            box-shadow: 0 12px 50px rgba(102, 126, 234, 0.5);
        }
    }
    
    .stButton > button:hover {
        animation: pulse 2s infinite;
    }
    
    /* ========== ИНФОРМАЦИОННЫЕ КАРТОЧКИ ========== */
    .info-card {
        background: linear-gradient(145deg, #1e1e1e, #2a2a2a);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .info-card:hover {
        border-color: #667eea;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
        transform: translateY(-4px);
    }
    
    .info-card h3 {
        color: #667eea;
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
    }
    
    .info-card p {
        color: #a0a0a0;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* ========== МЕТРИКИ ========== */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #667eea;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #a0a0a0;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem;
    }
    
    /* ========== ФОРМЫ И ИНПУТЫ ========== */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        background-color: #1e1e1e;
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-radius: 10px;
        color: white;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        background-color: #2a2a2a;
    }
    
    /* ========== КНОПКА ЗАКРЫТИЯ ========== */
    button[key="close_section"] {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 2px solid rgba(239, 68, 68, 0.3) !important;
        color: #ef4444 !important;
        padding: 0.5rem 1.5rem !important;
        height: auto !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    button[key="close_section"]:hover {
        background: rgba(239, 68, 68, 0.2) !important;
        border-color: #ef4444 !important;
        transform: scale(1.05) !important;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3) !important;
    }
    
    /* ========== РАЗДЕЛИТЕЛЬ ========== */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, rgba(102, 126, 234, 0.3), transparent);
        margin: 2.5rem 0;
    }
    
    /* ========== ТАБЛИЦЫ ========== */
    .dataframe {
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 8px;
    }
    
    /* ========== ПРОГРЕСС БАР ========== */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* ========== ЭКСПАНДЕРЫ ========== */
    .streamlit-expanderHeader {
        background-color: #1e1e1e;
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 8px;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #667eea;
        background-color: #2a2a2a;
    }
    
    /* ========== АЛЕРТЫ ========== */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid;
    }
    
    /* ========== ГРАДИЕНТНЫЙ ФОН ========== */
    .stApp {
        background: radial-gradient(circle at top right, rgba(102, 126, 234, 0.05), transparent),
                    radial-gradient(circle at bottom left, rgba(118, 75, 162, 0.05), transparent);
    }
    
    /* ========== ЗАГРУЗЧИК ФАЙЛОВ ========== */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        padding: 1rem;
        background: #1e1e1e;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: #2a2a2a;
    }
    
    /* ========== СТАТУС БЭДЖИ ========== */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .status-pending {
        background: rgba(251, 191, 36, 0.2);
        color: #fbbf24;
        border: 1px solid #fbbf24;
    }
    
    .status-running {
        background: rgba(59, 130, 246, 0.2);
        color: #3b82f6;
        border: 1px solid #3b82f6;
    }
    
    .status-completed {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
        border: 1px solid #22c55e;
    }
    
    .status-failed {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
    }
    
    /* ========== КАСТОМНЫЕ КАРТОЧКИ ========== */
    .custom-card {
        background: linear-gradient(145deg, #1e1e1e, #2a2a2a);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        border-color: #667eea;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
        transform: translateY(-2px);
    }
    
    /* ========== АНИМАЦИИ ========== */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animated {
        animation: fadeIn 0.5s ease-out;
    }
    /* Компактный вид датасетов */
    .compact-dataset-row {
        padding: 8px 0;
        border-bottom: 1px solid #333;
    }
    
    .compact-dataset-row:hover {
        background-color: rgba(255, 255, 255, 0.02);
    }
    
    /* Компактные метрики */
    div[data-testid="metric-container"] {
        padding: 0.5rem;
    }
    
    /* Статус-бейджи для паузированных задач */
    .status-badge.status-paused {
        background-color: #FFA500;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    
    /* Стили для мониторинга активных задач */
    .task-execution-box {
        background-color: rgba(0, 100, 255, 0.1);
        border-left: 3px solid #0064FF;
        padding: 8px;
        margin: 4px 0;
        border-radius: 4px;
    }
    
    .recent-execution-box {
        background-color: rgba(0, 255, 100, 0.05);
        border-left: 3px solid #00FF64;
        padding: 6px;
        margin: 4px 0;
        border-radius: 4px;
        font-size: 0.85rem;
    }
</style>
""",
        unsafe_allow_html=True,
    )
