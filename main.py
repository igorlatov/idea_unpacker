"""
Idea Unpacker — Main Orchestrator

A multi-model agentic flow for unpacking topics into compressed, insightful outputs.

Usage:
    python main.py

Requires API keys set as environment variables:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, DEEPSEEK_API_KEY
"""

import asyncio
from datetime import datetime
from schemas import UserInput, FlowResult
from steps import (
    step2_generate_ideas,
    step3_dual_scoring,
    step3b_select_top_idea,
    step5_format_and_criteria,
    step6_articulate,
    step7_evaluate,
    step7b_refine,
    step8_failure_analysis
)
import config


def log_provenance(provenance: list, step: str, model: str, detail: str):
    """Append to provenance trace."""
    provenance.append({
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "model": model,
        "detail": detail
    })


def get_user_input() -> UserInput:
    """Step 1: Capture user input via CLI."""
    print("\n" + "="*50)
    print("IDEA UNPACKER")
    print("="*50)
    
    topic = input("\nEnter topic (3-5 words): ").strip()
    intent = input("Enter your intent/lived experience (1 sentence): ").strip()
    
    return UserInput(topic=topic, intent=intent)


def display_ideas(scored_ideas, selected_index: int):
    """Display scored ideas for user review."""
    print("\n" + "-"*50)
    print("SCORED IDEAS (GPT + DeepSeek)")
    print("-"*50)
    
    for i, si in enumerate(scored_ideas):
        marker = "→ " if i == selected_index else "  "
        divergence = "⚡" if si.score_delta > config.SCORE_DIVERGENCE_THRESHOLD else ""
        print(f"{marker}{i+1}. {si.idea.name}")
        print(f"      Score: {si.combined_score:.1f} (δ={si.score_delta:.1f}) {divergence}")
        print(f"      Source: {si.idea.source}")
        print()


def user_checkpoint(scored_ideas, top_index: int) -> int:
    """Step 4: User confirms or changes selection."""
    display_ideas(scored_ideas, top_index)
    
    print(f"Selected: #{top_index + 1}")
    choice = input("Press Enter to confirm, or enter different number (1-{}): ".format(len(scored_ideas))).strip()
    
    if choice and choice.isdigit():
        new_index = int(choice) - 1
        if 0 <= new_index < len(scored_ideas):
            return new_index
    return top_index


async def run_flow() -> FlowResult:
    """Main orchestration loop."""
    provenance = []
    drafts = []
    evaluations = []
    score_history = []
    
    # Step 1: User Input
    user_input = get_user_input()
    log_provenance(provenance, "input", "user", f"topic={user_input.topic}")
    
    # Step 2: Generate Ideas
    print("\n⏳ Generating ideas via Claude...")
    ideas = await step2_generate_ideas(user_input)
    log_provenance(provenance, "ideation", "claude", f"generated {len(ideas)} ideas")
    
    # Step 3: Dual Scoring
    print("⏳ Scoring ideas (GPT + DeepSeek in parallel)...")
    scored_ideas = await step3_dual_scoring(ideas)
    top_idea = step3b_select_top_idea(scored_ideas)
    top_index = scored_ideas.index(top_idea)
    log_provenance(provenance, "scoring", "gpt+deepseek", f"top={top_idea.idea.name}")
    
    # Step 4: User Checkpoint
    final_index = user_checkpoint(scored_ideas, top_index)
    selected = scored_ideas[final_index]
    log_provenance(provenance, "selection", "user", f"confirmed={selected.idea.name}")
    
    # Step 5: Format & Criteria
    print("\n⏳ Designing format and criteria (DeepSeek)...")
    format_spec = await step5_format_and_criteria(selected, user_input)
    log_provenance(provenance, "format", "deepseek", 
                   f"type={format_spec.format_type.value}, bar={format_spec.minimum_bar}")
    
    print(f"\n📋 Format: {format_spec.format_type.value}")
    print(f"📊 Minimum bar: {format_spec.minimum_bar}")
    print(f"📏 Criteria: {', '.join(format_spec.criteria)}")
    
    # Step 6: Initial Articulation
    print("\n⏳ Creating initial draft (Claude)...")
    draft = await step6_articulate(selected, format_spec)
    drafts.append(draft)
    log_provenance(provenance, "draft_v1", "claude", f"words={draft.word_count}")
    
    # Step 7: Refinement Loop
    for cycle in range(config.MAX_REFINEMENT_CYCLES):
        print(f"\n⏳ Evaluation cycle {cycle + 1}/{config.MAX_REFINEMENT_CYCLES} (DeepSeek)...")
        
        evaluation = await step7_evaluate(draft, format_spec, score_history)
        evaluations.append(evaluation)
        score_history.append(evaluation.total_score)
        log_provenance(provenance, f"eval_v{draft.version}", "deepseek", 
                       f"score={evaluation.total_score}")
        
        print(f"   Score: {evaluation.total_score:.1f} / {format_spec.minimum_bar}")
        
        # Check success
        if evaluation.total_score >= format_spec.minimum_bar:
            print("✅ Bar met!")
            return FlowResult(
                success=True,
                final_draft=draft,
                final_score=evaluation.total_score,
                cycles_used=cycle + 1,
                provenance=provenance
            )
        
        # Check plateau
        if evaluation.plateau_detected:
            print("⚠️  Plateau detected — stopping early")
            break
        
        # Refine if not last cycle
        if cycle < config.MAX_REFINEMENT_CYCLES - 1:
            print(f"   Feedback: {evaluation.feedback[0][:60]}...")
            print(f"⏳ Refining (Claude)...")
            draft = await step7b_refine(draft, evaluation, format_spec, selected)
            drafts.append(draft)
            log_provenance(provenance, f"draft_v{draft.version}", "claude", 
                           f"words={draft.word_count}")
    
    # Step 8: Failure Path
    print("\n❌ Bar not met. Analyzing failure...")
    failure_reason = await step8_failure_analysis(drafts, evaluations, format_spec, selected)
    log_provenance(provenance, "failure_analysis", "deepseek", "completed")
    
    return FlowResult(
        success=False,
        final_draft=drafts[-1],
        final_score=score_history[-1],
        cycles_used=len(evaluations),
        failure_reason=failure_reason,
        provenance=provenance
    )


def display_result(result: FlowResult):
    """Display final output."""
    print("\n" + "="*50)
    if result.success:
        print("✅ SUCCESS")
    else:
        print("❌ DID NOT MEET BAR")
    print("="*50)
    
    print(f"\n📝 FINAL OUTPUT (v{result.final_draft.version}):\n")
    print(result.final_draft.content)
    print(f"\n💡 EXPLAINER:\n{result.final_draft.explainer}")
    
    print(f"\n📊 Final score: {result.final_score:.1f}")
    print(f"🔄 Cycles used: {result.cycles_used}")
    
    if result.failure_reason:
        print(f"\n🔍 FAILURE ANALYSIS:\n{result.failure_reason}")
    
    print("\n📜 PROVENANCE TRACE:")
    for p in result.provenance:
        print(f"   [{p['step']}] {p['model']}: {p['detail']}")


async def main():
    """Entry point."""
    try:
        result = await run_flow()
        display_result(result)
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
