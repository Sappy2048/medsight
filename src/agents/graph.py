"""
graph.py — LangGraph Orchestration Engine

Role:
    Wires all MedSight agents into a directed acyclic graph (DAG) with 
    conditional re-entry loops for quality control.
"""

import logging
from datetime import date
from typing import Optional, List, Dict, Any, TypedDict, Literal, Tuple
import asyncio

from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI
from qdrant_client import QdrantClient

from src.agents.copilot import preflight_validate, oversee_report, answer_question
from src.agents.resolution import resolve_prescription
from src.services.fda_client import get_past_and_present_labels
from src.agents.extraction import extract_interactions
from src.agents.temporal import compute_temporal_diff
from src.agents.impact import analyze_patient_impact
from src.agents.synthesis import synthesize_final_report

from src.schemas.prescription_schema import ParsedPrescription
from src.schemas.resolution_schema import ResolvedDrug
from src.schemas.diff_schema import DiffResult
from src.schemas.impact_schema import PatientImpactReport
from src.schemas.synthesizer_schema import MedSightFinalReport

logger = logging.getLogger("medsight.graph")

# ─── State Definition ─────────────────class MedSightState(TypedDict):
    # ── Input ─────────────────────────────────────
    raw_input:           str
    prescription:        Optional[ParsedPrescription]
    input_prescription_date: Optional[str]
    input_patient_age:       Optional[int]

    # ── Pipeline intermediates ────────────────────
    resolved_drugs:      List[ResolvedDrug]
    label_history:       Dict[str, Any] 
    # extraction_results: { generic_name -> {"past": ExtractionResult, "present": ExtractionResult} }
    extraction_results:  Dict[str, Any]
    diffs:               List[DiffResult]
    reasoning:           List[Dict[str, Any]]
    impact_report:       Optional[PatientImpactReport]

    # ── Output ────────────────────────────────────
    final_report:        Optional[MedSightFinalReport]

    # ── Copilot control ───────────────────────────
    copilot_session:     List[Dict[str, str]]
    loop_count:          int
    awaiting_input:      bool
    should_rerun:        bool  # <-- Track routing decisions cleanly in state
    clarification_message: Optional[str]

    # ── Error tracking ────────────────────────────
    errors:              List[str]


# ─── Node Functions ───────────────────────────────────────────────────────────

def create_nodes(llm_client: AsyncOpenAI, qdrant_client: QdrantClient, db_pool: Any):
    
    async def copilot_preflight_node(state: MedSightState) -> Dict[str, Any]:
        logger.info("Node: copilot_preflight")
        prescription, clarification_msg = await preflight_validate(
            state["raw_input"], 
            llm_client,
            override_date=state.get("input_prescription_date"),
            override_age=state.get("input_patient_age")
        )
        
        return {
            "prescription": prescription,
            "clarification_message": clarification_msg,
            "awaiting_input": bool(clarification_msg)
        }�───────────────────

def create_nodes(llm_client: AsyncOpenAI, qdrant_client: QdrantClient, db_pool: Any):
    
    async def copilot_preflight_node(state: MedSightState) -> Dict[str, Any]:
        logger.info("Node: copilot_preflight")
        prescription, clarification_msg = await preflight_validate(state["raw_input"], llm_client)
        
        return {
            "prescription": prescription,
            "clarification_message": clarification_msg,
            "awaiting_input": bool(clarification_msg)
        }

    async def resolver_node(state: MedSightState) -> Dict[str, Any]:
        logger.info("Node: resolver")
        if state["prescription"] is None:
            raise ValueError("Prescription is None — preflight validation failed or returned empty.")
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            resolved_drugs = await resolve_prescription(state["prescription"], http_client)
        return {"resolved_drugs": resolved_drugs}

    async def label_fetcher_node(state: MedSightState) -> Dict[str, Any]:
        logger.info("Node: label_fetcher (Concurrent + FDC Aware)")
        if state["prescription"] is None:
            raise ValueError("Prescription is None — cannot fetch labels.")
        import httpx
        from src.services.fda_client import get_past_and_present_labels
        
        raw_date = state["prescription"].prescription_date
        if raw_date is None:
            prescription_date = "2026-06-01"
        elif isinstance(raw_date, date):
            prescription_date = raw_date.isoformat()
        else:
            prescription_date = str(raw_date)
        
        label_history = {}
        
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            # Concurrent worker path per constituent generic
            async def fetch_worker(generic_name: str) -> Tuple[str, Optional[Tuple[Any, Any]]]:
                try:
                    past, present = await get_past_and_present_labels(
                        generic_name, prescription_date, http_client
                    )
                    return generic_name, (past, present)
                except Exception as e:
                    logger.error(f"Failed to fetch labels for constituent {generic_name}: {e}")
                    return generic_name, None

            # SPEED TWEAK + FDC FIX: Extract EVERY constituent generic across ALL drugs into a set.
            # The set automatically deduplicates common salts across different prescribed products!
            all_individual_generics = {
                g_name for drug in state["resolved_drugs"] for g_name in drug.generic_names
            }

            # HIGH CONCURRENCY: Fan out all label fetches simultaneously using connection pooling
            tasks = [fetch_worker(g_name) for g_name in all_individual_generics]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for res in results:
                if isinstance(res, BaseException) or res is None:
                    continue
                generic_name, label_pair = res
                if label_pair:
                    label_history[generic_name] = label_pair
                    
        return {"label_history": label_history}

    async def temporal_node(state: MedSightState) -> Dict[str, Any]:
        logger.info("Node: temporal (Concurrent Extractions + Concurrent Matrix Cross-Diffs)")
        if state["prescription"] is None:
            raise ValueError("Prescription is None — cannot compute temporal diff.")
        
        raw_date = state["prescription"].prescription_date
        if raw_date is None:
            prescription_date_str = None
        elif isinstance(raw_date, date):
            prescription_date_str = raw_date.isoformat()
        else:
            prescription_date_str = str(raw_date)
        
        all_generics = []
        for drug in state["resolved_drugs"]:
            all_generics.extend(drug.generic_names)

        diffs = []
        reasoning_list = []
        extraction_results: Dict[str, Any] = {}
        
        # Concurrent workflow execution for an isolated source drug
        async def process_source_generic(
            source_generic: str, past_label: Any, present_label: Any
        ) -> Tuple[str, Any, Any, List[Tuple[Any, Any]]]:
            # SPEED TWEAK: Extract past and present labels simultaneously
            past_task = extract_interactions(past_label, source_generic, llm_client)
            present_task = extract_interactions(present_label, source_generic, llm_client)
            past_ext, present_ext = await asyncio.gather(past_task, present_task)
            
            other_generics = [g for g in all_generics if g != source_generic]
            
            # SPEED TWEAK: Fan out the entire combination target comparison matrix simultaneously
            diff_tasks = [
                compute_temporal_diff(past_ext, present_ext, target, llm_client, prescription_date_str)
                for target in other_generics
            ]
            diff_results = await asyncio.gather(*diff_tasks)
            # Return extraction results alongside diffs so state can surface them
            return source_generic, past_ext, present_ext, list(diff_results)

        # HIGH CONCURRENCY: Loop natively scales across every resolved constituent salt from the history dictionary
        source_tasks = [
            process_source_generic(source_generic, past_label, present_label)
            for source_generic, (past_label, present_label) in state["label_history"].items()
        ]
        
        source_results = await asyncio.gather(*source_tasks, return_exceptions=True)
        
        for result in source_results:
            if isinstance(result, BaseException):
                logger.error(f"Source generic processing chunk failed: {result}")
                continue
            source_generic, past_ext, present_ext, diff_list = result
            extraction_results[source_generic] = {"past": past_ext, "present": present_ext}
            for diff, reasoning in diff_list:
                diffs.append(diff)
                reasoning_list.append(reasoning)
                
        return {"diffs": diffs, "reasoning": reasoning_list, "extraction_results": extraction_results}

    async def impact_node(state: MedSightState) -> Dict[str, Any]:
        logger.info("Node: impact")
        if state["prescription"] is None:
            raise ValueError("Prescription is None — cannot analyze impact.")
        
        diff_tuples = list(zip(state["diffs"], state["reasoning"]))
        raw_date = state["prescription"].prescription_date
        if raw_date is None:
            prescription_date_str = "2024-01-01"
        elif isinstance(raw_date, date):
            prescription_date_str = raw_date.isoformat()
        else:
            prescription_date_str = str(raw_date)
        
        impact_report = await analyze_patient_impact(
            diffs=diff_tuples,
            resolved_drugs=state["resolved_drugs"],
            prescription_date=prescription_date_str,
            llm_client=llm_client,
            qdrant_client=qdrant_client
        )
        return {"impact_report": impact_report}

    async def synthesizer_node(state: MedSightState) -> Dict[str, Any]:
        logger.info("Node: synthesizer")
        if state["impact_report"] is None:
            raise ValueError("Impact report is None — cannot synthesize final report.")
        final_report = await synthesize_final_report(
            report=state["impact_report"],
            diff_results=state["diffs"],
            llm_client=llm_client
        )
        return {"final_report": final_report}

    async def copilot_overseer_node(state: MedSightState) -> Dict[str, Any]:
        logger.info("Node: copilot_overseer")
        if state["final_report"] is None:
            raise ValueError("Final report is None — cannot oversee report.")
        
        # Call overseer_report EXACTLY ONCE here
        agent_should_rerun, explanation = await oversee_report(state["final_report"], llm_client)
        
        new_loop_count = state["loop_count"]
        actual_should_rerun = False

        # Apply loop protection guardrails cleanly
        if agent_should_rerun and state["loop_count"] < 2:
            new_loop_count += 1
            actual_should_rerun = True
            logger.warning(f"Overseer requesting re-run. Loop count: {new_loop_count}")
        
        return {
            "loop_count": new_loop_count, 
            "awaiting_input": False,
            "should_rerun": actual_should_rerun
        }

    async def persist_node(state: MedSightState) -> Dict[str, Any]:
        logger.info("Node: persist")
        errors = []
        try:
            logger.info("Saving report to database...")
        except Exception as e:
            errors.append(f"Database save failed: {e}")
            
        return {"errors": errors}

    return {
        "copilot_preflight": copilot_preflight_node,
        "resolver": resolver_node,
        "label_fetcher": label_fetcher_node,
        "temporal": temporal_node,
        "impact": impact_node,
        "synthesizer": synthesizer_node,
        "copilot_overseer": copilot_overseer_node,
        "persist": persist_node
    }

# ─── Graph Construction ───────────────────────────────────────────────────────

def build_medsight_graph(llm_client: AsyncOpenAI, qdrant_client: QdrantClient, db_pool: Any):
    nodes = create_nodes(llm_client, qdrant_client, db_pool)
    
    workflow = StateGraph(MedSightState)
    
    for name, func in nodes.items():
        workflow.add_node(name, func)
        
    workflow.set_entry_point("copilot_preflight")
    
    def preflight_router(state: MedSightState) -> str:
        if state.get("awaiting_input"):
            logger.info("Graph halted: Awaiting user clarification.")
            return END
        return "resolver"

    workflow.add_conditional_edges(
        "copilot_preflight",
        preflight_router
    )
    
    workflow.add_edge("resolver", "label_fetcher")
    workflow.add_edge("label_fetcher", "temporal")
    workflow.add_edge("temporal", "impact")
    workflow.add_edge("impact", "synthesizer")
    workflow.add_edge("synthesizer", "copilot_overseer")
    
    # Simple, stateless router relying entirely on the state context flag
    def should_continue(state: MedSightState) -> str:
        if state.get("should_rerun"):
            return "resolver"
        return "persist"
        
    workflow.add_conditional_edges(
        "copilot_overseer",
        should_continue,
        {
            "resolver": "resolver",
            "persist": "persist"
        }
    )
    
    workflow.add_edge("persist", END)
    
    return workflow.compile()

# ─── Public Entry Points ──────────────────────────────────────────────────────

async def run_medsight(
    raw_input: str,
    llm_client: AsyncOpenAI,
    qdrant_client: QdrantClient,
    db_pool: Any,
) -> Dict[str, Any]:
    """
    Main entry point for the MedSight pipeline.
    """
    app = build_medsight_graph(llm_client, qdrant_client, db_pool)
    
    initial_state = MedSightState(
        raw_input=raw_input,
        prescription=None,
        resolved_drugs=[],
        label_history={},
        extraction_results={},
        diffs=[],
        reasoning=[],
        impact_report=None,
        final_report=None,
        copilot_session=[],
        loop_count=0,
        awaiting_input=False,
        should_rerun=False,
        clarification_message=None,
        errors=[]
    )
    
    final_state = await app.ainvoke(initial_state)

    if final_state.get("awaiting_input"):
        return {
            "status": "clarification_required",
            "message": final_state["clarification_message"],
            "partial_prescription": final_state["prescription"]
        }
    
    if final_state["final_report"] is None:
        raise RuntimeError("MedSight pipeline failed to generate a final report.")
        
    return {
        "status": "success",
        "report": final_state["final_report"]
    }

async def run_copilot_qa(
    question: str,
    report: MedSightFinalReport,
    history: List[Dict[str, str]],
    llm_client: AsyncOpenAI,
) -> str:
    """
    Separate entry point for grounded Q&A mode.
    """
    return await answer_question(question, report, history, llm_client)