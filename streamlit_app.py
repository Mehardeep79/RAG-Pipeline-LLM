import streamlit as st
import numpy as np
import wikipedia
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import re

# Page configuration
st.set_page_config(
    page_title=" RAG Pipeline Prototype For LLMs",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'document' not in st.session_state:
    st.session_state.document = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None

# Cache functions for better performance
@st.cache_resource
def load_models():
    """Load and cache the ML models"""
    with st.spinner("🤖 Loading AI models..."):
        # Load embedding model
        embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        
        # Load QA model
        qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
        qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)
        
        return embedding_model, qa_pipeline

@st.cache_data
def get_wikipedia_content(topic):
    """Fetch Wikipedia content with caching"""
    try: 
        page = wikipedia.page(topic)
        return page.content, None
    except wikipedia.exceptions.PageError:
        return None, "Page not found. Please try a different topic."
    except wikipedia.exceptions.DisambiguationError as e:
        return None, f"Ambiguous topic. Try one of these: {', '.join(e.options[:5])}"

def split_text(text, chunk_size=256, chunk_overlap=20):
    """Split text into overlapping chunks"""
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    
    # Split into sentences first
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        test_chunk = current_chunk + ". " + sentence if current_chunk else sentence
        test_tokens = tokenizer.tokenize(test_chunk)
        
        if len(test_tokens) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                
                # Add overlap
                if chunk_overlap > 0 and chunks:
                    overlap_tokens = tokenizer.tokenize(current_chunk)
                    if len(overlap_tokens) > chunk_overlap:
                        overlap_start = len(overlap_tokens) - chunk_overlap
                        overlap_text = tokenizer.convert_tokens_to_string(overlap_tokens[overlap_start:])
                        current_chunk = overlap_text + ". " + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk = sentence
        else:
            current_chunk = test_chunk
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def create_visualizations(chunks, distances, answer):
    """Create interactive visualizations"""
    
    # 1. Chunk length distribution
    chunk_lengths = [len(chunk.split()) for chunk in chunks]
    
    fig_chunks = make_subplots(
        rows=1, cols=2,
        subplot_titles=("📏 Chunk Length Distribution", "📊 Statistical Summary"),
        specs=[[{"type": "bar"}, {"type": "box"}]]
    )
    
    # Histogram
    fig_chunks.add_trace(
        go.Histogram(x=chunk_lengths, nbinsx=15, name="Distribution",
                    marker_color="skyblue", opacity=0.7),
        row=1, col=1
    )
    
    # Box plot
    fig_chunks.add_trace(
        go.Box(y=chunk_lengths, name="Statistics", 
               marker_color="lightgreen", boxmean=True),
        row=1, col=2
    )
    
    fig_chunks.update_layout(height=400, showlegend=False)
    
    # 2. Similarity scores
    if distances is not None:
        similarity_scores = 1 / (1 + distances[0])
        
        fig_similarity = go.Figure(data=[
            go.Bar(x=[f"Rank {i+1}" for i in range(len(similarity_scores))],
                   y=similarity_scores,
                   marker_color=['gold', 'silver', '#CD7F32'][:len(similarity_scores)],
                   text=[f'{score:.3f}' for score in similarity_scores],
                   textposition='auto')
        ])
        
        fig_similarity.update_layout(
            title="🎯 Retrieved Chunks Similarity Scores",
            xaxis_title="Retrieved Chunk Rank",
            yaxis_title="Similarity Score",
            height=400
        )
    else:
        fig_similarity = None
    
    # 3. Confidence gauge
    if answer:
        confidence = answer['score']
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "🎯 Answer Confidence (%)"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 20], 'color': "red"},
                    {'range': [20, 40], 'color': "orange"},
                    {'range': [40, 60], 'color': "yellow"},
                    {'range': [60, 80], 'color': "lightgreen"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig_gauge.update_layout(height=400)
    else:
        fig_gauge = None
    
    return fig_chunks, fig_similarity, fig_gauge

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 RAG Pipeline Prototype For LLMs 🚀</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
        An intelligent Q&A system powered by 🤗 Hugging Face, 📖 Wikipedia, and ⚡ FAISS vector search
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    embedding_model, qa_pipeline = load_models()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    chunk_size = st.sidebar.slider("📏 Chunk Size (tokens)", 128, 512, 256, 32)
    chunk_overlap = st.sidebar.slider("🔗 Chunk Overlap (tokens)", 10, 50, 20, 5)
    k_retrieval = st.sidebar.slider("🔍 Chunks to Retrieve", 1, 10, 3, 1)
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip**: Larger chunks provide more context but may be less precise. More retrieval chunks give broader context.")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📖 Choose Your Knowledge Source")
        topic = st.text_input(
            "Enter a Wikipedia topic:",
            placeholder="e.g., Artificial Intelligence, Climate Change, Python Programming",
            help="Enter any topic available on Wikipedia"
        )
        
        if st.button("🔄 Fetch & Process Article", type="primary"):
            if topic:
                # Fetch content
                with st.spinner("📖 Fetching Wikipedia article..."):
                    document, error = get_wikipedia_content(topic)
                
                if error:
                    st.error(f"❌ {error}")
                    return
                
                if document:
                    st.session_state.document = document
                    
                    # Process text
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("✂️ Splitting text into chunks...")
                    progress_bar.progress(25)
                    chunks = split_text(document, chunk_size, chunk_overlap)
                    st.session_state.chunks = chunks
                    
                    status_text.text("🧮 Creating embeddings...")
                    progress_bar.progress(50)
                    embeddings = embedding_model.encode(chunks)
                    st.session_state.embeddings = embeddings
                    
                    status_text.text("⚡ Building FAISS index...")
                    progress_bar.progress(75)
                    dimension = embeddings.shape[1]
                    index = faiss.IndexFlatL2(dimension)
                    index.add(np.array(embeddings))
                    st.session_state.index = index
                    st.session_state.embedding_model = embedding_model
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")
                    
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"✅ Successfully processed '{topic}' into {len(chunks)} chunks!")
            else:
                st.warning("⚠️ Please enter a topic first!")
    
    with col2:
        if st.session_state.chunks:
            st.metric("📚 Total Chunks", len(st.session_state.chunks))
            st.metric("🧮 Embedding Dimension", st.session_state.embeddings.shape[1])
            st.metric("📝 Average Chunk Length", f"{np.mean([len(chunk.split()) for chunk in st.session_state.chunks]):.1f} words")
    
    # Q&A Section
    if st.session_state.chunks:
        st.markdown("---")
        st.header("❓ Ask Your Question")
        
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the main concept? How does it work?",
            help="Ask any question about the processed article"
        )
        
        if st.button("🎯 Get Answer", type="primary") and question:
            with st.spinner("🔍 Searching for relevant information..."):
                # Get query embedding
                query_embedding = st.session_state.embedding_model.encode([question])
                
                # Search
                distances, indices = st.session_state.index.search(np.array(query_embedding), k_retrieval)
                retrieved_chunks = [st.session_state.chunks[i] for i in indices[0]]
                
                # Generate answer
                context = " ".join(retrieved_chunks)
                answer = qa_pipeline(question=question, context=context)
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🤖 Generated Answer")
                st.success(f"**{answer['answer']}**")
                
                confidence = answer['score']
                if confidence >= 0.8:
                    confidence_color = "green"
                    confidence_text = "Very High"
                elif confidence >= 0.6:
                    confidence_color = "blue"
                    confidence_text = "High"
                elif confidence >= 0.4:
                    confidence_color = "orange"
                    confidence_text = "Medium"
                else:
                    confidence_color = "red"
                    confidence_text = "Low"
                
                st.markdown(f"**Confidence:** :{confidence_color}[{confidence:.1%} ({confidence_text})]")
            
            with col2:
                st.subheader("📊 Quick Stats")
                st.metric("🎯 Confidence Score", f"{confidence:.1%}")
                st.metric("📝 Answer Length", f"{len(answer['answer'])} chars")
                st.metric("🔍 Chunks Used", len(retrieved_chunks))
            
            # Retrieved chunks
            st.subheader("📋 Retrieved Context Chunks")
            for i, chunk in enumerate(retrieved_chunks):
                with st.expander(f"📄 Chunk {i+1} (Similarity: {1/(1+distances[0][i]):.3f})"):
                    st.write(chunk)
            
            # Visualizations
            st.markdown("---")
            st.header("📊 Pipeline Analytics")
            
            fig_chunks, fig_similarity, fig_gauge = create_visualizations(
                st.session_state.chunks, distances, answer
            )
            
            tab1, tab2, tab3 = st.tabs(["📏 Chunk Analysis", "🎯 Similarity Scores", "📊 Confidence Meter"])
            
            with tab1:
                st.plotly_chart(fig_chunks, use_container_width=True)
                
            with tab2:
                if fig_similarity:
                    st.plotly_chart(fig_similarity, use_container_width=True)
                else:
                    st.info("Run a query first to see similarity scores!")
                    
            with tab3:
                if fig_gauge:
                    st.plotly_chart(fig_gauge, use_container_width=True)
                else:
                    st.info("Run a query first to see confidence metrics!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🔍 RAG Pipeline Demo | Built with ❤️ using Streamlit, Hugging Face, and FAISS</p>
        <p>🤗 Models: sentence-transformers/all-mpnet-base-v2 | deepset/roberta-base-squad2</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
