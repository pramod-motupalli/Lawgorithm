from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from lawgorithm.agents import RouterAgent, CaseParsingAgent, RetrievalAgent, VerdictAgent
from lawgorithm import config
import os

app = FastAPI(title="Lawgorithm API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Model
class CaseRequest(BaseModel):
    plaintiff: str
    defendant: str
    description: str
    api_key: str = None

# Initialize Agents
router = RouterAgent()
parser = CaseParsingAgent()
retriever = RetrievalAgent()
verdict_agent = VerdictAgent()

@app.post("/predict")
async def predict_verdict(request: CaseRequest):
    try:
        # Set API Key if provided
        if request.api_key:
            os.environ["GROQ_API_KEY"] = request.api_key
            config.GROQ_API_KEY = request.api_key
        
        if not config.GROQ_API_KEY:
            raise HTTPException(status_code=400, detail="Groq API Key is required")

        # Step 1: Routing
        category = router.route(request.description)
        
        # Step 2: Parsing
        case_structure = parser.parse(request.description, request.plaintiff, request.defendant)
        
        # Step 3: Retrieval
        precedents = retriever.retrieve(case_structure['summary'], category)
        
        # Step 4: Verdict
        raw_verdict = verdict_agent.predict(case_structure, precedents, category)
        
        # Parse Outcome
        lines = raw_verdict.split('\n')
        outcome = "Unknown"
        verdict_text = raw_verdict
        
        # Iterate through first few lines to find the outcome
        for i, line in enumerate(lines[:5]):
            if "IN FAVOUR OF:" in line.upper():
                outcome = line.split(":", 1)[1].strip()
                # Remove the outcome line from the text
                verdict_text = "\n".join(lines[i+1:]).strip()
                break
            
        return {
            "category": category,
            "outcome": outcome,
            "verdict_text": verdict_text,
            "case_structure": case_structure,
            "precedents": precedents
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "Lawgorithm API is running"}
