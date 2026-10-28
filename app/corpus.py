import os
import re
import json
import hashlib
from typing import List, Dict, Optional, Any
import fitz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

SECTION_MAP = {
    "1": "Introduction", "2": "Admission", "3": "Academic Session",
    "4": "Course Structure", "5": "Institute dues payment and semester registration",
    "6": "Attendance and Leave", "7": "Assessment of Performance",
    "8": "Examinations", "9": "Graduation Requirement",
    "10": "Award of Degree", "11": "Change of Branch",
    "APPENDIX-I": "Alternative mid-semester and supplementary examinations, summer courses and backlog papers",
    "APPENDIX-II": "Policy on sanction of medical leave to students",
    "APPENDIX-III": "Composition of co-ordination committee for UG courses",
    "APPENDIX-IV": "Guidelines for award of letter grades",
    "APPENDIX-V": "Grading system",
    "APPENDIX-VI": "Administration of short term industrial or research experience"
}

PENALTY_THRESHOLDS = {
    "regular": {
        "3-1-0": {"S": 4, "no_penalty_max_A": 8,  "grade_cut_max_A": 16, "debar_A": 16, "debar_TA": 24, "clause": "6.2.5"},
        "3-0-0": {"S": 3, "no_penalty_max_A": 6,  "grade_cut_max_A": 12, "debar_A": 12, "debar_TA": 18, "clause": "6.2.5"},
        "0-0-3": {"S": 1, "no_penalty_max_A": 2,  "grade_cut_max_A": 4,  "debar_A": 4,  "debar_TA": 6,  "clause": "6.2.5"},
    },
    "summer": {
        "3-1-0": {"S": 3,   "no_penalty_max_A": 6, "grade_cut_max_A": 12, "debar_A": 12, "debar_TA": 18, "clause": "6.2.5"},
        "3-0-0": {"S": 2.5, "no_penalty_max_A": 5, "grade_cut_max_A": 10, "debar_A": 10, "debar_TA": 15, "clause": "6.2.5"},
        "0-0-3": {"S": 1,   "no_penalty_max_A": 2, "grade_cut_max_A": 3,  "debar_A": 4,  "debar_TA": 6,  "clause": "6.2.5"},
    }
}

class NITRCorpusTools:
    def __init__(self, pdf_path: str = "backUp.pdf", index_dir: str = "index_store"):
        self.pdf_path = pdf_path
        self.index_dir = index_dir
        self.model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedder = SentenceTransformer(self.model_name)
        self.chunks: List[Dict] = []
        self.clause_lookup: Dict[str, Dict] = {}
        self.index: Optional[faiss.IndexFlatIP] = None
        self._init_index()

    def _compute_hash(self) -> str:
        h = hashlib.sha256()
        if os.path.exists(self.pdf_path):
            with open(self.pdf_path, "rb") as f:
                h.update(f.read())
        h.update(self.model_name.encode())
        return h.hexdigest()[:10]

    def _parse_pdf(self) -> List[Dict]:
        doc = fitz.open(self.pdf_path)
        clause_re = re.compile(r"^(\d+\.\d+(?:\.\d+)?)\s+(.*)")
        app_re = re.compile(r"^(APPENDIX-[IVX]+)\s*$", re.IGNORECASE)
        app_item_re = re.compile(r"^(\d+)\.\s+(.*)")

        chunks, curr_app, curr = [], None, None
        for p_idx in range(2, len(doc)):
            p_num = p_idx + 1
            for raw in doc[p_idx].get_text("text").split("\n"):
                line = raw.strip()
                if not line or (line.startswith("Page ") and " of 31" in line):
                    continue
                m_app = app_re.match(line)
                if m_app:
                    if curr: chunks.append(curr)
                    curr_app = m_app.group(1).upper()
                    curr = {"clause_id": curr_app, "parent_section": curr_app,
                            "section_title": SECTION_MAP.get(curr_app, "Appendix"),
                            "page_number": p_num, "content": line}
                    continue
                if not curr_app:
                    m_c = clause_re.match(line)
                    if m_c:
                        if curr: chunks.append(curr)
                        cid = m_c.group(1)
                        curr = {"clause_id": cid, "parent_section": ".".join(cid.split(".")[:-1]),
                                "section_title": SECTION_MAP.get(cid.split(".")[0], "General"),
                                "page_number": p_num, "content": line}
                    elif curr:
                        curr["content"] += " " + line
                else:
                    m_item = app_item_re.match(line)
                    if m_item:
                        if curr: chunks.append(curr)
                        cid = f"{curr_app}.{m_item.group(1)}"
                        curr = {"clause_id": cid, "parent_section": curr_app,
                                "section_title": SECTION_MAP.get(curr_app, "Appendix"),
                                "page_number": p_num, "content": f"[{cid}] {line}"}
                    elif curr:
                        curr["content"] += " " + line
        if curr: chunks.append(curr)
        return chunks

    def _init_index(self):
        os.makedirs(self.index_dir, exist_ok=True)
        v_hash = self._compute_hash()
        idx_path = os.path.join(self.index_dir, f"nitr_{v_hash}.faiss")
        meta_path = os.path.join(self.index_dir, f"nitr_{v_hash}.json")

        if os.path.exists(idx_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(idx_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
        else:
            self.chunks = self._parse_pdf()
            texts = [f"{c['section_title']} | {c['clause_id']}: {c['content']}" for c in self.chunks]
            embs = self.embedder.encode(texts, normalize_embeddings=True)
            self.index = faiss.IndexFlatIP(embs.shape[1])
            self.index.add(np.array(embs, dtype="float32"))
            faiss.write_index(self.index, idx_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, indent=2)

        self.clause_lookup = {c["clause_id"]: c for c in self.chunks}

    # --- AGENT TOOLS ---
    def search_corpus(self, query: str, top_k: int = 4) -> List[Dict]:
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(np.array(q_emb, dtype="float32"), top_k)
        return [
            {**self.chunks[i], "score": round(float(s), 4)}
            for s, i in zip(scores[0], idxs[0]) if i != -1
        ]

    def get_clause(self, clause_id: str) -> Optional[Dict]:
        return self.clause_lookup.get(clause_id.strip().upper())

    def get_petition_fields(self, petition_dict: Dict) -> Dict:
        return {
            "ltp": petition_dict["ltp"],
            "semester_type": petition_dict["semester_type"],
            "raw_absences": petition_dict["raw_absences"],
            "medical_leave_days": petition_dict["medical_leave_days"],
            "medical_sanctioned": petition_dict["medical_sanctioned"]
        }

    def lookup_penalty_table(self, ltp: str, raw_absences: int, medical_leave_days: int,
                             medical_sanctioned: bool, semester_type: str = "regular") -> Dict[str, Any]:
        sem = semester_type.lower()
        if sem not in PENALTY_THRESHOLDS or ltp not in PENALTY_THRESHOLDS[sem]:
            return {"status": "OUT_OF_TABLE", "route_hint": "escalate", "clauses": ["6.2.8"]}

        rules = PENALTY_THRESHOLDS[sem][ltp]
        B = medical_leave_days if medical_sanctioned else 0
        A = max(0, raw_absences - B)
        TA = raw_absences

        if TA > rules["debar_TA"]:
            return {"outcome": "X_GRADE_TA_EXCEEDED", "A": A, "TA": TA, "route_hint": "reject",
                    "clauses": ["6.2.1", "6.2.5"]}
        if A > rules["debar_A"]:
            return {"outcome": "X_GRADE_A_EXCEEDED", "A": A, "TA": TA, "route_hint": "reject",
                    "clauses": ["6.2.1", "6.2.5"]}
        if A > rules["no_penalty_max_A"]:
            return {"outcome": "GRADE_CUT_CONDONATION_BAND", "A": A, "TA": TA, "route_hint": "refer",
                    "clauses": ["6.2.1", "6.2.3", "6.2.5"]}
        return {"outcome": "ROUTINE_CONDONATION_BAND", "A": A, "TA": TA, "route_hint": "grant-under-provision",
                "clauses": ["6.2.2", "6.2.5"]}