import gradio as gr
import numpy as np
import wikipedia
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Global variables to store models and data
embedding_model = None
qa_pipeline = None
chunks = None
embeddings = None
index = None
document = None

def load_models():
    """Load and cache the ML models"""
    global embedding_model, qa_pipeline
    
    if embedding_model is None:
        print("🤖 Loading embedding model...")
        embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        
        print("🤖 Loading QA model...")
        qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
        qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)
        
        print("✅ Models loaded successfully!")
    
    return "✅ Models are ready!"

def get_wikipedia_content(topic):
    """Fetch Wikipedia content"""
    try: 
        page = wikipedia.page(topic)
        return page.content, f"✅ Successfully fetched '{topic}' article"
    except wikipedia.exceptions.PageError:
        return None, f"❌ Page '{topic}' not found. Please try a different topic."
    except wikipedia.exceptions.DisambiguationError as e:
        return None, f"⚠️ Ambiguous topic. Try one of these: {', '.join(e.options[:5])}"

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

def process_article(topic, chunk_size, chunk_overlap):
    """Process Wikipedia article into chunks and embeddings"""
    global chunks, embeddings, index, document
    
    if not topic.strip():
        return "⚠️ Please enter a topic first!", None, ""
    
    # Load models first
    load_models()
    
    # Fetch content
    document, message = get_wikipedia_content(topic)
    
    if document is None:
        return message, None, ""
    
    # Process text
    chunks = split_text(document, int(chunk_size), int(chunk_overlap))
    
    # Create embeddings
    embeddings = embedding_model.encode(chunks)
    
    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    # Create summary stats
    chunk_lengths = [len(chunk.split()) for chunk in chunks]
    summary = f"""
📊 **Processing Summary:**
- **Total chunks**: {len(chunks)}
- **Embedding dimension**: {dimension}
- **Average chunk length**: {np.mean(chunk_lengths):.1f} words
- **Min/Max chunk length**: {min(chunk_lengths)}/{max(chunk_lengths)} words
- **Document length**: {len(document.split())} words

✅ Ready for questions!
"""
    
    return f"✅ Successfully processed '{topic}' into {len(chunks)} chunks!", create_chunk_visualization(), summary

def create_chunk_visualization():
    """Create chunk length distribution plot"""
    if chunks is None:
        return None
    
    chunk_lengths = [len(chunk.split()) for chunk in chunks]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("📏 Chunk Length Distribution", "📊 Statistical Summary"),
        specs=[[{"type": "bar"}, {"type": "box"}]]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=chunk_lengths, nbinsx=15, name="Distribution",
                    marker_color="skyblue", opacity=0.7),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(y=chunk_lengths, name="Statistics", 
               marker_color="lightgreen", boxmean=True),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False, title="📊 Chunk Analysis")
    
    return fig

def answer_question(question, k_retrieval):
    """Answer question using RAG pipeline"""
    global chunks, embeddings, index, qa_pipeline
    
    if chunks is None or index is None:
        return "⚠️ Please process an article first!", None, "", ""
    
    if not question.strip():
        return "⚠️ Please enter a question!", None, "", ""
    
    # Get query embedding
    query_embedding = embedding_model.encode([question])
    
    # Search
    distances, indices = index.search(np.array(query_embedding), int(k_retrieval))
    retrieved_chunks = [chunks[i] for i in indices[0]]
    
    # Generate answer
    context = " ".join(retrieved_chunks)
    answer = qa_pipeline(question=question, context=context)
    
    # Format results
    confidence = answer['score']
    
    # Determine confidence level
    if confidence >= 0.8:
        confidence_emoji = "🟢"
        confidence_text = "Very High"
    elif confidence >= 0.6:
        confidence_emoji = "🔵"
        confidence_text = "High"
    elif confidence >= 0.4:
        confidence_emoji = "🟡"
        confidence_text = "Medium"
    else:
        confidence_emoji = "🔴"
        confidence_text = "Low"
    
    # Format answer
    formatted_answer = f"""
🤖 **Answer**: {answer['answer']}

{confidence_emoji} **Confidence**: {confidence:.1%} ({confidence_text})
📏 **Answer Length**: {len(answer['answer'])} characters
🔍 **Chunks Used**: {len(retrieved_chunks)}
"""
    
    # Format retrieved chunks
    retrieved_text = "📋 **Retrieved Context Chunks:**\n\n"
    for i, chunk in enumerate(retrieved_chunks):
        similarity = 1 / (1 + distances[0][i])
        retrieved_text += f"**Chunk {i+1}** (Similarity: {similarity:.3f}):\n{chunk}\n\n---\n\n"
    
    # Create similarity visualization
    similarity_scores = 1 / (1 + distances[0])
    similarity_plot = create_similarity_plot(similarity_scores)
    
    return formatted_answer, similarity_plot, retrieved_text, create_confidence_gauge(confidence)

def create_similarity_plot(similarity_scores):
    """Create similarity scores bar chart"""
    fig = go.Figure(data=[
        go.Bar(x=[f"Rank {i+1}" for i in range(len(similarity_scores))],
               y=similarity_scores,
               marker_color=['gold', 'silver', '#CD7F32'][:len(similarity_scores)],
               text=[f'{score:.3f}' for score in similarity_scores],
               textposition='auto')
    ])
    
    fig.update_layout(
        title="🎯 Retrieved Chunks Similarity Scores",
        xaxis_title="Retrieved Chunk Rank",
        yaxis_title="Similarity Score",
        height=400
    )
    
    return fig

def create_confidence_gauge(confidence):
    """Create confidence gauge visualization"""
    fig = go.Figure(go.Indicator(
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
    
    fig.update_layout(height=400)
    return fig

def clear_data():
    """Clear all processed data"""
    global chunks, embeddings, index, document
    chunks = None
    embeddings = None
    index = None
    document = None
    return "🗑️ Data cleared! Ready for new article.", None, "", "", None, None, ""

# Create Gradio interface
def create_interface():
    """Create the main Gradio interface"""
    
    with gr.Blocks(
        title="🔍 RAG Pipeline For LLMs",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        """
    ) as interface:
        
        # Header
        gr.Markdown("""
        # 🔍 RAG Pipeline Prototype For LLMs 🚀
        
        <div style="text-align: center; color: #666; margin-bottom: 2rem;">
        An intelligent Q&A system powered by 🤗 Hugging Face, 📖 Wikipedia, and ⚡ FAISS vector search
        </div>
        """)
        
        with gr.Tab("📖 Article Processing"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📋 Step 1: Configure & Process Article")
                    
                    topic_input = gr.Textbox(
                        label="📖 Wikipedia Topic",
                        placeholder="e.g., Artificial Intelligence, Climate Change, Python Programming",
                        info="Enter any topic available on Wikipedia"
                    )
                    
                    with gr.Row():
                        chunk_size = gr.Slider(
                            label="📏 Chunk Size (tokens)",
                            minimum=128,
                            maximum=512,
                            value=256,
                            step=32,
                            info="Larger chunks = more context, smaller chunks = more precision"
                        )
                        
                        chunk_overlap = gr.Slider(
                            label="🔗 Chunk Overlap (tokens)",
                            minimum=10,
                            maximum=50,
                            value=20,
                            step=5,
                            info="Overlap helps maintain context between chunks"
                        )
                    
                    process_btn = gr.Button("🔄 Fetch & Process Article", variant="primary", size="lg")
                    
                    processing_status = gr.Textbox(
                        label="📊 Processing Status",
                        interactive=False
                    )
                    
                with gr.Column(scale=1):
                    processing_summary = gr.Markdown("### 📈 Processing Summary\n*Process an article to see statistics*")
            
            chunk_plot = gr.Plot(label="📊 Chunk Analysis Visualization")
            
        with gr.Tab("❓ Question Answering"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 🎯 Step 2: Ask Your Question")
                    
                    question_input = gr.Textbox(
                        label="❓ Your Question",
                        placeholder="e.g., What is the main concept? How does it work?",
                        info="Ask any question about the processed article"
                    )
                    
                    k_retrieval = gr.Slider(
                        label="🔍 Number of Chunks to Retrieve",
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        info="More chunks = broader context, fewer chunks = more focused"
                    )
                    
                    answer_btn = gr.Button("🎯 Get Answer", variant="primary", size="lg")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 💡 Tips\n- Process an article first\n- Ask specific questions\n- Adjust retrieval count for better results")
            
            answer_output = gr.Markdown(label="🤖 Generated Answer")
            
            with gr.Row():
                similarity_plot = gr.Plot(label="🎯 Similarity Scores")
                confidence_gauge = gr.Plot(label="📊 Confidence Meter")
        
        with gr.Tab("📋 Retrieved Context"):
            retrieved_chunks = gr.Markdown(
                label="📄 Retrieved Chunks",
                value="*Ask a question to see retrieved context chunks*"
            )
        
        with gr.Tab("🔧 System Controls"):
            gr.Markdown("### 🛠️ System Management")
            
            with gr.Row():
                load_models_btn = gr.Button("🤖 Load Models", variant="secondary")
                clear_btn = gr.Button("🗑️ Clear Data", variant="stop")
            
            system_status = gr.Textbox(
                label="🔧 System Status",
                value="Ready to load models...",
                interactive=False
            )
        
        # Event handlers
        process_btn.click(
            fn=process_article,
            inputs=[topic_input, chunk_size, chunk_overlap],
            outputs=[processing_status, chunk_plot, processing_summary]
        )
        
        answer_btn.click(
            fn=answer_question,
            inputs=[question_input, k_retrieval],
            outputs=[answer_output, similarity_plot, retrieved_chunks, confidence_gauge]
        )
        
        load_models_btn.click(
            fn=load_models,
            outputs=[system_status]
        )
        
        clear_btn.click(
            fn=clear_data,
            outputs=[processing_status, chunk_plot, processing_summary, answer_output, similarity_plot, confidence_gauge, retrieved_chunks]
        )
        
        # Footer
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #666; padding: 1rem;">
        🔍 RAG Pipeline Demo | Built with ❤️ using Gradio, Hugging Face, and FAISS<br>
        🤗 Models: sentence-transformers/all-mpnet-base-v2 | deepset/roberta-base-squad2
        </div>
        """)
    
    return interface

# Launch the app
if __name__ == "__main__":
    import sys
    
    # Check if share is requested via command line
    enable_share = "--share" in sys.argv
    
    app = create_interface()
    
    print("🚀 Starting RAG Pipeline Gradio Interface...")
    print("📖 Models will load when first needed")
    if enable_share:
        print("🌐 Public sharing ENABLED")
        print("⚠️  If antivirus blocks this, restart without --share")
    else:
        print("📱 Running locally only (safer for antivirus)")
        print("💡 Add --share argument to enable public links")
    
    app.launch(
        server_name="127.0.0.1" if not enable_share else "0.0.0.0",
        server_port=7860,
        share=enable_share,
        show_error=True,
        debug=False,
        inbrowser=True
    )
