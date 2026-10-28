import time
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from app.corpus import NITRCorpusTools
from app.schemas import CitationItem, sanitize_output_rationale

class AgentState(TypedDict):
    petition: Dict[str, Any]
    category: str
    extracted_entities: Dict[str, Any]
    retrieved_clauses: Dict[str, Dict]
    table_evaluation: Dict[str, Any]
    candidate_citations: List[str]
    verified_citations: List[CitationItem]
    route: str
    referred_to: List[str]
    rationale: str
    steps: int
    max_steps: int
    start_time: float

class PetitionTriageGraph:
    def __init__(self, tools: NITRCorpusTools, max_steps: int = 6):
        self.tools = tools
        self.max_steps = max_steps
        self.graph = self._build_graph()

    def _triage_node(self, state: AgentState) -> AgentState:
        """Agent 1: Classify petition intent and extract structured parameters."""
        p = state["petition"]
        text_lower = p["petition_text"].lower()
        fields = self.tools.get_petition_fields(p)

        # Out-of-corpus check (e.g., hostel / mess / roommate disputes)
        if any(w in text_lower for w in ["hostel", "mess", "roommate", "warden", "food quality"]):
            category = "out_of_corpus"
        elif any(w in text_lower for w in ["catastrophic", "exceptional", "senate", "beyond standard"]):
            category = "exceptional_discretion"
        else:
            category = "attendance_and_leave"

        return {
            **state,
            "category": category,
            "extracted_entities": fields,
            "steps": state["steps"] + 1
        }

    def _regulation_analyst_node(self, state: AgentState) -> AgentState:
        """
        Agent 2: Multi-hop retrieval.
        Hop 1: Retrieve base attendance rules & evaluate penalty table.
        Hop 2: If medical leave or condonation applies, re-search modifying clauses (Appendix II / 6.2.3).
        """
        if state["category"] == "out_of_corpus":
            return {
                **state,
                "route": "escalate",
                "referred_to": ["Academic Section Officer (Out of Corpus)"],
                "candidate_citations": [],
                "rationale": "Petition topic falls outside the NIT Rourkela B.Tech Academic Regulations corpus.",
                "steps": state["steps"] + 1
            }

        p = state["petition"]
        ent = state["extracted_entities"]
        retrieved: Dict[str, Dict] = {}

        # Hop 1: Semantic search on raw petition + Penalty Table evaluation
        for hit in self.tools.search_corpus(p["petition_text"], top_k=3):
            retrieved[hit["clause_id"]] = hit

        table_res = self.tools.lookup_penalty_table(
            ltp=ent["ltp"],
            raw_absences=ent["raw_absences"],
            medical_leave_days=ent["medical_leave_days"],
            medical_sanctioned=ent["medical_sanctioned"],
            semester_type=ent["semester_type"]
        )

        for cid in table_res.get("clauses", []):
            clause_obj = self.tools.get_clause(cid)
            if clause_obj:
                retrieved[cid] = clause_obj

        # Hop 2: Re-search modifying provisions if medical leave or exceptional discretion is involved
        referred_to = []
        candidate_cids = list(table_res.get("clauses", []))

        if ent["medical_leave_days"] > 0:
            med_hits = self.tools.search_corpus("Policy on sanction of medical leave Dean SW prescription of rest", top_k=3)
            for hit in med_hits:
                retrieved[hit["clause_id"]] = hit
            for app_cid in ["APPENDIX-II.3", "APPENDIX-II.10"]:
                c_obj = self.tools.get_clause(app_cid)
                if c_obj:
                    retrieved[app_cid] = c_obj
                    if app_cid not in candidate_cids:
                        candidate_cids.append(app_cid)

        # Route & Authority Determination
        if state["category"] == "exceptional_discretion" or table_res.get("status") == "OUT_OF_TABLE":
            for esc_cid in ["6.2.8", "APPENDIX-II.11"]:
                c_obj = self.tools.get_clause(esc_cid)
                if c_obj:
                    retrieved[esc_cid] = c_obj
                    candidate_cids.append(esc_cid)
            route = "escalate"
            referred_to = ["Senate", "Director"]
            rationale = (
                f"Total absences ({ent['raw_absences']}) or circumstances exceed standard condonation tables. "
                "Escalated under Clause 6.2.8 and Appendix II.11 reserved discretion."
            )
        else:
            route = table_res["route_hint"]
            if route == "refer":
                referred_to.append("Dean (AA)")
                if ent["medical_leave_days"] > 0 and ent["medical_sanctioned"]:
                    referred_to.append("Dean (SW)")
                rationale = (
                    f"Raw absences ({table_res['TA']}) exceed 12, which would trigger an X grade under 6.2.1/6.2.5. "
                    f"However, deducting {ent['medical_leave_days']} sanctioned medical leave days (Appendix II.3) "
                    f"yields {table_res['A']} unauthorized absences (within the 7-12 band for a {ent['ltp']} course) "
                    f"and {table_res['TA']} total absences (within the 18 limit). Referred to {', '.join(referred_to)} "
                    "for discretionary one-step grade cut condonation under Clause 6.2.3."
                )
            elif route == "reject":
                rationale = (
                    f"Effective unauthorized absences ({table_res['A']}) or total absences ({table_res['TA']}) "
                    f"exceed maximum permissible thresholds for {ent['ltp']} under Clause 6.2.5. Mandatory X grade applies under Clause 6.2.1."
                )
            else:
                rationale = (
                    f"Unauthorized absences ({table_res['A']}) are within the 15% routine condonation threshold "
                    f"for {ent['ltp']} under Clauses 6.2.2 and 6.2.5."
                )

        return {
            **state,
            "retrieved_clauses": retrieved,
            "table_evaluation": table_res,
            "candidate_citations": list(dict.fromkeys(candidate_cids)),
            "route": route,
            "referred_to": referred_to,
            "rationale": sanitize_output_rationale(rationale),
            "steps": state["steps"] + 1
        }

    def _verifier_node(self, state: AgentState) -> AgentState:
        """
        Agent 3 (Verifier): Every cited clause ID must resolve to retrieved corpus text.
        If any citation fails verification or step limit is exceeded, escalate immediately.
        """
        if state["steps"] >= state["max_steps"]:
            return {
                **state,
                "route": "escalate",
                "rationale": "Agent loop step limit exceeded. Escalating incomplete run to human officer."
            }

        verified_list: List[CitationItem] = []
        all_resolved = True

        for cid in state["candidate_citations"]:
            chunk = state["retrieved_clauses"].get(cid) or self.tools.get_clause(cid)
            if chunk and chunk.get("content"):
                verified_list.append(
                    CitationItem(
                        clause_id=cid,
                        page_number=chunk["page_number"],
                        snippet=chunk["content"][:180] + "...",
                        verified=True
                    )
                )
            else:
                all_resolved = False

        if not all_resolved:
            return {
                **state,
                "route": "escalate",
                "verified_citations": verified_list,
                "rationale": "Citation verification failed to resolve one or more clauses against retrieved text. Escalating."
            }

        return {
            **state,
            "verified_citations": verified_list,
            "steps": state["steps"] + 1
        }

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("triage", self._triage_node)
        workflow.add_node("analyst", self._regulation_analyst_node)
        workflow.add_node("verifier", self._verifier_node)

        workflow.set_entry_point("triage")
        workflow.add_edge("triage", "analyst")
        workflow.add_edge("analyst", "verifier")
        workflow.add_edge("verifier", END)
        return workflow.compile()

    def run_petition(self, petition_dict: Dict[str, Any]) -> AgentState:
        initial_state: AgentState = {
            "petition": petition_dict,
            "category": "",
            "extracted_entities": {},
            "retrieved_clauses": {},
            "table_evaluation": {},
            "candidate_citations": [],
            "verified_citations": [],
            "route": "escalate",
            "referred_to": [],
            "rationale": "",
            "steps": 0,
            "max_steps": self.max_steps,
            "start_time": time.time()
        }
        return self.graph.invoke(initial_state)