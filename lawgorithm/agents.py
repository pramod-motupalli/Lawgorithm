import json
import numpy as np
from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
import faiss
from . import config
from . import utils

from langchain_groq import ChatGroq

class BaseAgent:
    def __init__(self):
        # Enforce Groq
        self.llm = ChatGroq(
            model=config.LLM_MODEL_NAME,
            api_key=config.GROQ_API_KEY,
            temperature=0.2
        )

class RouterAgent(BaseAgent):
    def route(self, case_description: str) -> str:
        """
        Determines if the case is Criminal, Civil, or Traffic.
        """
        prompt = PromptTemplate.from_template(
            """
            You are a legal expert. Classify the following case description into exactly one of these categories:
            - Criminal
            - Civil
            - Traffic

            Case Description: {description}

            Return ONLY the category name.
            """
        )
        chain = prompt | self.llm
        result = chain.invoke({"description": case_description})
        return result.content.strip()

class CaseParsingAgent(BaseAgent):
    def parse(self, description: str, plaintiff: str, defendant: str) -> Dict[str, Any]:
        """
        Structures the raw case details into a legal summary.
        """
        prompt = PromptTemplate.from_template(
            """
            You are a legal case analyst. Analyze the following case details and structure them.

            Plaintiff/Petitioner: {plaintiff}
            Defendant/Respondent: {defendant}
            Raw Description: {description}

            Output a valid JSON object with the following keys:
            - "summary": A professional legal summary of the facts.
            - "legal_issues": A list of core legal issues or offenses.
            - "key_facts": A list of key factual points.
            - "statutes_involved": A list of potential Acts or Sections involved (inferred).

            Ensure the output is pure JSON without markdown formatting.
            """
        )
        chain = prompt | self.llm
        try:
            result = chain.invoke({
                "description": description,
                "plaintiff": plaintiff,
                "defendant": defendant
            })
            content = result.content.strip()
            # Clean up markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            return json.loads(content)
        except Exception as e:
            print(f"Error parsing case: {e}")
            return {
                "summary": description,
                "legal_issues": [],
                "key_facts": [],
                "statutes_involved": []
            }

from .retriever import HybridRetriever

class RetrievalAgent:
    def __init__(self):
        self.retriever = HybridRetriever()

    def retrieve(self, query: str, category: str, k: int = 3) -> List[Dict[str, Any]]:
        return self.retriever.search(query, category, k)

class VerdictAgent(BaseAgent):
    def predict(self, case_structure: Dict[str, Any], similar_cases: List[Dict[str, Any]], category: str) -> str:
        """
        Generates the verdict prediction.
        """
        # Format similar cases for the prompt
        precedents_text = ""
        for i, case in enumerate(similar_cases, 1):
            precedents_text += f"\n--- Precedent {i} ---\n"
            precedents_text += f"Title: {case.get('title')}\n"
            precedents_text += f"Verdict: {case.get('verdict')}\n"
            precedents_text += f"Summary: {case.get('judgment_summary')[:500]}...\n"

        prompt = PromptTemplate.from_template(
            """
            You are a Legal Output Correction Agent responsible for validating and correcting AI-generated legal predictions.
            Your job is to ensure that every output is accurate, grounded in facts, legally consistent, and free from hallucinations.

            ### CURRENT CASE FACTS
            Summary: {summary}
            Legal Issues: {legal_issues}
            Key Facts: {key_facts}

            ### SIMILAR PAST PRECEDENTS
            {precedents}

            ### STRICT RULES
            1. **No Hallucinations**: Do NOT invent IPC sections, CrPC sections, Acts, evidence, or precedents. Remove anything not supported by facts.
            2. **Only Apply IPC/Acts that Match Facts**: Include only legal sections justified by the case facts (e.g., Injury -> IPC 323/325, Theft -> IPC 379). Remove unsupported sections.
            3. **Maintain Factual Consistency**: Stay strictly within given facts. Do not add new events or witnesses.
            4. **Correct Legal Reasoning**: Logic must be step-by-step and based only on facts. Avoid emotional or speculative statements.
            5. **No Fake Precedents**: If real precedents are not provided, refer generally to court requirements.
            6. **Keep Output Minimal & Relevant**: Probable verdict, relevant sections, concise reasoning only.
            7. **Neutral, Academic Tone**: Use probabilistic language ("likely", "probable"). Do NOT give legal advice.
            8. **Mandatory Disclaimer**: End with: "This is only a probabilistic academic prediction based on the provided facts. It is not legal advice or an official court verdict."

            ### FORMATTING INSTRUCTIONS
            - **OUTCOME LINE**: The VERY FIRST line MUST be exactly: "IN FAVOUR OF: [Plaintiff/Defendant/Both/None]".
            - **HIGHLIGHT SECTIONS**: Highlight mentioned IPC sections/Acts in bold (e.g., **Section 302 IPC**).
            - **DO NOT INCLUDE**: Case No, Date, Court Name, Judge's Name.

            ### YOUR CORRECTED LEGAL OUTPUT
            """
        )
        
        chain = prompt | self.llm
        result = chain.invoke({
            "summary": case_structure.get("summary"),
            "legal_issues": ", ".join([str(x) for x in case_structure.get("legal_issues", [])]),
            "key_facts": ", ".join([str(x) for x in case_structure.get("key_facts", [])]),
            "precedents": precedents_text
        })
        
        return result.content
