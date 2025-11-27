
import streamlit as st
import sys
import os
import time

# Ajouter le chemin pour importer tes modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

from dj_ia_complet import CompleteDJSystem, Agent1_MusicCurator, Agent2_AudioPlayer, Agent3_AudienceAnalyzer

def main():
    st.set_page_config(
        page_title="DJ IA Intelligent",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 Ton DJ IA Intelligent")
    st.markdown("Système de mixage automatique avec intelligence artificielle")
    
    # Sidebar pour la configuration
    st.sidebar.title("🎛️ Configuration")
    
    contexte = st.sidebar.selectbox(
        "Contexte de la session",
        ["soiree", "diner", "afterwork", "relax"],
        index=0
    )
    
    duree = st.sidebar.slider("Durée (minutes)", 1, 60, 10)
    
    # Boutons de contrôle
    col1, col2 = st.sidebar.columns(2)
    with col1:
        demarrer = st.button("🎧 Démarrer", type="primary")
    with col2:
        arreter = st.button("⏹️ Arrêter")
    
    # Zone principale
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎪 Session en Cours")
        
        if demarrer:
            with st.spinner("Lancement du DJ IA..."):
                # Initialisation du système
                systeme_dj = CompleteDJSystem()
                
                # Zone de progression
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulation de la session
                status_text.text("🎵 Préparation de la playlist...")
                time.sleep(1)
                
                status_text.text("🎧 Mixage des morceaux...")
                time.sleep(1)
                
                status_text.text("🎤 Analyse de l'audience...")
                time.sleep(1)
                
                # Lance la session
                morceaux_joues = systeme_dj.adaptive_dj_session(contexte, duree)
                
                progress_bar.progress(100)
                status_text.text(f"✅ Session terminée! {morceaux_joues} morceaux joués.")
                
                st.balloons()
                st.success("Session DJ IA terminée avec succès!")
    
    with col2:
        st.subheader("📊 Métriques")
        
        # Métriques simulées
        st.metric("Énergie Audience", "85%", "+5%")
        st.metric("Engagement", "78%", "+3%")
        st.metric("Morceaux Joués", "12")
        st.metric("Humeur", "Joyeux")
        
        st.subheader("🎛️ Contrôles")
        st.slider("Volume", 0, 100, 80)
        st.selectbox("Effets", ["Reverb", "Echo", "Flanger", "Aucun"])
    
    # Section d'information
    st.markdown("---")
    st.subheader("🤖 Tes Agents IA")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**🎵 Agent 1**\nSélection musicale intelligente")
    with col2:
        st.info("**🎧 Agent 2**\nMixage automatique")
    with col3:
        st.info("**🎤 Agent 3**\nAnalyse d'audience")

if __name__ == "__main__":
    main()
