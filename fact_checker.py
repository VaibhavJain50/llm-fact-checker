import json
import spacy
import chromadb
import pandas as pd
import google.generativeai as genai
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# Initialize NLP model
nlp = spacy.load("en_core_web_sm")

class FactChecker:
    def __init__(self, api_key: str):
        """
        Initializes Google GenAI, Embedding Model, and Vector DB.
        """
        # 1. Configure Google Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # 2. Load Local Embedding Model (No API cost for embeddings)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 3. Setup Vector Database
        self.chroma_client = chromadb.Client()
        
        try:
            self.chroma_client.delete_collection("fact_check_db")
        except:
            pass
        self.collection = self.chroma_client.create_collection(name="fact_check_db")

    def load_knowledge_base(self, csv_path: str):
        try:
            df = pd.read_csv(csv_path)
            documents = df['text'].tolist()
            ids = [str(i) for i in df['id'].tolist()]
            
            # Embed locally
            embeddings = self.embedding_model.encode(documents).tolist()
            
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids
            )
            return True, f"Loaded {len(documents)} facts into ChromaDB."
        except Exception as e:
            return False, str(e)

    def extract_entities(self, text: str) -> List[str]:
        doc = nlp(text)
        return [ent.text for ent in doc.ents]

    def check_claim(self, claim: str):
        # Step 1: Retrieve
        query_embedding = self.embedding_model.encode([claim]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=3
        )
        
        retrieved_facts = results['documents'][0]
        distances = results['distances'][0]
        
        # Step 2: Prompt Gemini
        prompt = f"""
        Act as an objective Fact Checker.
        
        CLAIM: "{claim}"
        
        KNOWN FACTS:
        {json.dumps(retrieved_facts)}
        
        Task:
        1. Check if the claim contradicts or is supported by the KNOWN FACTS.
        2. Output valid JSON only.
        
        Format:
        {{
            "verdict": "True" | "False" | "Unverifiable",
            "reasoning": "One short sentence explaining why.",
            "evidence_used": ["Quote the exact fact used"]
        }}
        """
        
        # Step 3: Generate
        response = self.model.generate_content(prompt)
        
        # Robust Parsing (Clean markdown if Gemini adds it)
        clean_text = response.text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:-3]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:-3]
            
        try:
            result_json = json.loads(clean_text)
            # Attach metadata
            result_json['retrieved_context'] = retrieved_facts
            result_json['entities_detected'] = self.extract_entities(claim)
            result_json['confidence_scores'] = [round(1 - d, 2) for d in distances]
            return result_json
        except json.JSONDecodeError:
            return {
                "verdict": "Error", 
                "reasoning": "Model output format error", 
                "evidence_used": [], 
                "raw_response": response.text
            }